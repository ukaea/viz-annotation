from toktagger.api.schemas.projects import Project
from toktagger.api.schemas.samples import Sample
from toktagger.api.schemas.annotations import TimePoint
from toktagger.api.schemas.data import TimeSeriesData, DataParams
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from toktagger.api.schemas.annotations import Annotation
import typing
from toktagger.api.models.base import Model, ModelRegistry
import logging
import pydantic

logger = logging.getLogger("ray")


class DisruptionCNNTrainParams(pydantic.BaseModel):
    train_val_test_split: list[typing.Annotated[float, pydantic.Field(gt=0, lt=1)]] = (
        pydantic.Field(
            min_length=3,
            max_length=3,
            description="Fraction of the total annotations to use in the training / validation / test sets. Fractions should sum to 1.",
        )
    )

    num_epochs: int = pydantic.Field(gt=0, default=100)
    batch_size: int = pydantic.Field(gt=0, default=32)
    patience: int = pydantic.Field(
        gt=0,
        default=20,
        description="If no improvement in accuracy is seen after this number of epochs, training will be stopped early.",
    )
    threshold: float = pydantic.Field(
        gt=0,
        default=1e-4,
        description="Threshold over which a new epoch is considered to have improved in accuracy.",
    )
    device: typing.Literal["cpu", "cuda", "mps", "xpu"] = "cpu"

    @pydantic.field_validator("train_val_test_split", mode="after")
    @classmethod
    def validate_sum_to_one(cls, value: list[float]) -> list[float]:
        if sum(value) != 1:
            raise ValueError("Train / Val / Test fractions must sum to 1!")
        return value


class DisruptionCNNPredictParams(pydantic.BaseModel):
    batch_size: int = pydantic.Field(gt=0, default=32)
    device: typing.Literal["cpu", "cuda", "mps", "xpu"] = "cpu"


class DisruptionDataset(Dataset):  # Inherit from torch.utils.dataset
    def __init__(
        self,
        project: Project,
        samples: list[Sample],
        annotations: list[list[Annotation]],
        data_loader: DataLoader,
    ):
        self.data_loader = data_loader
        self.samples = samples
        self.annotations = annotations
        self.current_scaling = []
        self.time_scaling = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.tensor, int]:
        data: TimeSeriesData = self.data_loader.get_sample(
            self.samples[idx], DataParams()
        ).values["ip"]
        # Scale data to be between 0 and 1...
        plasma_current = torch.tensor(data.values, dtype=torch.float32)
        self.current_scaling.append(plasma_current.max())
        plasma_current = plasma_current / plasma_current.max()
        self.time_scaling.append(data.time[-1])
        if not self.annotations:
            return plasma_current

        annotation: TimePoint = self.annotations[idx][0]
        disruption_time = annotation.time
        return plasma_current, disruption_time / data.time[-1]


@ModelRegistry.register(
    "disruption_cnn",
    ["time-series"],
    DisruptionCNNTrainParams,
    DisruptionCNNPredictParams,
)
class DisruptionCNN(Model):
    def define_model(self):
        return nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def train(
        self,
        samples: list[Sample],
        annotations: list[list[Annotation]],
        params: DisruptionCNNTrainParams,
    ) -> float:
        self.log_progress(training_status="started")

        if not self.gpu_available() and params.device != "cpu":
            raise ValueError(
                "Only CPU available on current worker node, but non-CPU device requested!"
            )

        device = torch.device(params.device)
        self.model.to(device)

        # Remove any samples which don't have annotations
        new_samples = []
        new_annotations = []

        for sample, annotation in zip(samples, annotations):
            if annotation:
                new_samples.append(sample)
                new_annotations.append(annotation)

        samples = new_samples
        annotations = new_annotations

        self.split_data(
            samples=samples,
            annotations=annotations,
            train_val_test_split=params.train_val_test_split,
        )

        self.train_dataset = DisruptionDataset(
            self.project, self.train_samples, self.train_annotations, self.data_loader
        )
        self.val_dataset = (
            DisruptionDataset(
                self.project, self.val_samples, self.val_annotations, self.data_loader
            )
            if self.val_samples
            else None
        )
        self.test_dataset = (
            DisruptionDataset(
                self.project, self.test_samples, self.test_annotations, self.data_loader
            )
            if self.test_samples
            else None
        )
        train_loader = DataLoader(
            self.train_dataset, batch_size=params.batch_size, shuffle=False
        )
        val_loader = (
            DataLoader(self.val_dataset, batch_size=params.batch_size, shuffle=False)
            if self.val_dataset
            else None
        )
        test_loader = (
            DataLoader(self.test_dataset, batch_size=params.batch_size, shuffle=False)
            if self.test_dataset
            else None
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        if not self.train_samples or not self.train_annotations:
            raise ValueError("No samples or annotations found!")

        loss_history = {"train": [], "val": []}
        best_val_loss = float("inf")
        print_per_epoch = max(1, params.num_epochs // 50)  # print 50 times

        train_accuracy = None
        val_accuracy = None
        test_accuracy = None

        for epoch in range(params.num_epochs):
            self.model.train()
            total_train_loss = 0
            sum_correct = 0
            sum_total = 0
            y_train_true, y_train_pred = [], []

            for batch_samples, batch_annotations in train_loader:
                batch_samples = batch_samples.unsqueeze(1)
                batch_samples = batch_samples.to(device)
                batch_annotations = batch_annotations.float().to(device)

                outputs = self.model(batch_samples).squeeze(1)
                loss = criterion(outputs, batch_annotations)
                error = torch.abs(outputs - batch_annotations)

                correct = error <= 0.05 * batch_annotations
                sum_correct += correct.sum().item()
                sum_total += batch_samples.size(0)
                train_accuracy = (sum_correct / sum_total) * 100

                total_train_loss += loss.item()
                y_train_true.extend(batch_annotations.cpu().numpy())
                y_train_pred.extend(outputs.detach().cpu().numpy())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss_avg = total_train_loss / len(self.train_dataset)
            train_mae = mean_absolute_error(y_train_true, y_train_pred)
            train_rmse = root_mean_squared_error(y_train_true, y_train_pred)

            # --- Validation phase ---
            if val_loader:
                self.model.eval()
                total_val_loss = 0
                sum_correct = 0
                sum_total = 0
                y_val_true, y_val_pred = [], []

                with torch.no_grad():
                    for batch_samples, batch_annotations in val_loader:
                        batch_samples = batch_samples.unsqueeze(1)
                        batch_samples = batch_samples.to(device)
                        batch_annotations = batch_annotations.float().to(device)

                        outputs = self.model(batch_samples).squeeze(1)
                        loss = criterion(outputs, batch_annotations)
                        error = torch.abs(outputs - batch_annotations)

                        correct = error <= 0.05 * batch_annotations
                        sum_correct += correct.sum().item()
                        sum_total += batch_samples.size(0)
                        val_accuracy = (sum_correct / sum_total) * 100

                        total_val_loss += loss.item()
                        y_val_true.extend(batch_annotations.cpu().numpy())
                        y_val_pred.extend(outputs.cpu().numpy())

                val_loss_avg = total_val_loss / len(val_loader)
                val_mae = mean_absolute_error(y_val_true, y_val_pred)
                val_rmse = root_mean_squared_error(y_val_true, y_val_pred)

                loss_history["train"].append(train_loss_avg)
                loss_history["val"].append(val_loss_avg)

                self.log_progress(
                    progress=int((epoch / params.num_epochs) * 100), score=val_accuracy
                )

                if epoch % print_per_epoch == 0:
                    logger.debug(f"Epoch [{epoch + 1}/{params.num_epochs}]")
                    logger.debug(
                        f"  Train Loss: {train_loss_avg:.4f}, MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}"
                    )
                    logger.debug(
                        f"  Val   Loss: {val_loss_avg:.4f}, MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}"
                    )

            # --- Early stopping ---
            if val_loss_avg < best_val_loss - params.threshold:
                best_val_loss = val_loss_avg
                epochs_since_improvement = 0
                self.best_model = self.model
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= params.patience:
                    logger.debug(
                        f"No validation improvement for {params.patience} epochs. Stopping early."
                    )
                    self.model = self.best_model
                    break

        # --- Evaluation on test set ---
        if test_loader:
            self.model.eval()
            sum_correct = 0
            sum_total = 0

            with torch.no_grad():
                for batch_samples, batch_annotations in test_loader:
                    batch_samples = batch_samples.unsqueeze(1)
                    batch_samples = batch_samples.to(device)
                    batch_annotations = batch_annotations.float().to(device)

                    outputs = self.model(batch_samples).squeeze(1)

                    error = torch.abs(outputs - batch_annotations)

                    correct = error <= 0.05 * batch_annotations
                    sum_correct += correct.sum().item()
                    sum_total += batch_samples.size(0)
                    test_accuracy = (sum_correct / sum_total) * 100

            logger.debug(f"Test Accuracy: {(sum_correct / sum_total) * 100}")

        final_accuracy = test_accuracy or val_accuracy or train_accuracy

        self.log_progress(
            training_status="completed",
            progress=int((epoch / params.num_epochs) * 100),
            score=final_accuracy,
        )

        return final_accuracy

    def predict(
        self,
        samples: list[Sample],
        params: DisruptionCNNPredictParams,
        data_params: DataParams | None = None,
    ) -> list[list[TimePoint]]:
        if not self.gpu_available() and params.device != "cpu":
            raise ValueError(
                "Only CPU available on current worker node, but non-CPU device requested!"
            )

        device = torch.device(params.device)
        self.model.to(device)
        num_mc_samples = 20  # Should let user choose num mc samples? TODO
        dataset = DisruptionDataset(
            self.project, samples, annotations=None, data_loader=self.data_loader
        )

        self.model.train()  # Using dropout so has to be in train mode
        all_predictions: list[list[torch.tensor]] = []
        dataloader = DataLoader(dataset, batch_size=params.batch_size, shuffle=False)

        for i in range(num_mc_samples):
            predictions: list[torch.tensor] = []
            with torch.no_grad():
                for batch_samples in dataloader:
                    batch_samples = batch_samples.unsqueeze(1)
                    batch_samples = batch_samples.to(device)
                    predictions.append(self.model(batch_samples))

            all_predictions.append(torch.cat(predictions, dim=0))

        stacked_predictions = torch.stack(all_predictions)
        # Because we've done 50x mc samples, just use the first lot of scaling values...
        scaling = torch.tensor(
            dataset.time_scaling[: int(len(dataset.time_scaling) / num_mc_samples)],
            device=device,
        ).squeeze()
        means = stacked_predictions.mean(dim=0).squeeze(dim=1) * scaling
        stds = stacked_predictions.std(dim=0).squeeze(dim=1) * scaling
        return [
            [
                TimePoint(
                    validated=False,
                    uncertainty=stds[i].item(),
                    label="Disruption",
                    time=means[i].item(),
                    created_by=self.type,
                )
            ]
            for i in range(len(samples))
        ]

    def save(self, file_stem: str):
        torch.save(self.model.state_dict(), f"{file_stem}.model")

    def load(self, file_path: str):
        self.model.load_state_dict(torch.load(file_path, map_location="cpu"))
