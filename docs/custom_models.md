# Custom Models

Models in TokTagger are machine learning components that can be trained on annotated data to make predictions on new samples. They integrate with the annotation workflow to provide automated labeling capabilities. TokTagger comes with built-in models for common tasks, but you can easily create custom models for your specific use cases.

!!! tip
    Note that you will need to install extra dependencies into your environment to enable the ML Models functionality:
    ```
    pip install toktagger[models]
    ```

## Overview

A model is responsible for:

1. **Defining the model architecture** (e.g., neural network, classical ML algorithm)
2. **Training on annotated data** with train/validation/test splits
3. **Making predictions** on new samples with uncertainty estimates
4. **Saving and loading** trained model weights

## Creating a Custom Model

### Step 1: Import Required Components

```python
from toktagger.api.models.base import Model, ModelRegistry
from toktagger.api.schemas.samples import Sample
from toktagger.api.schemas.annotations import Annotation, AnnotationIn
from toktagger.api.schemas.projects import Project, Task
import typing
```

### Step 2: Define Your Model Class

Create a class that inherits from `Model` and implement the required methods:

```python
@ModelRegistry.register("my_custom_model", ["time-series"])
class MyCustomModel(Model):
    """Custom model for my specific task"""
    
    def define_model(self):
        """
        Define and return your model architecture.
        
        Returns:
            The model object (e.g., PyTorch nn.Module, TensorFlow model, scikit-learn estimator)
        """
        # Your model architecture here
        pass
    
    def train(
        self,
        samples: list[Sample],
        annotations: list[list[Annotation]],
        params: pydantic.BaseModel | None = None
    ) -> float:
        """
        Train the model on annotated samples.
        
        Args:
            samples: List of samples to train on
            annotations: List of annotation lists (one list per sample)
            params: An optional user defined Pydantic model containing parameters needed when training
            
        Returns:
            Final model accuracy/score metric
        """
        # Your training logic here
        pass
    
    def predict(
        self, 
        samples: list[Sample],
        params: pydantic.BaseModel | None = None
        ) -> list[list[AnnotationIn]]:
        """
        Make predictions on new samples.
        
        Args:
            samples: List of samples to predict on
            params: An optional user defined Pydantic model containing parameters needed when predicting
            
        Returns:
            List of annotation lists (predictions for each sample)
        """
        # Your prediction logic here
        pass
    
    def save(self, file_path: str):
        """
        Save the trained model to disk.
        
        Args:
            file_path: Path where model should be saved
        """
        # Your save logic here
        pass
    
    def load(self, file_path: str):
        """
        Load a trained model from disk.
        
        Args:
            file_path: Path to the saved model
        """
        # Your load logic here
        pass
```

### Step 3: Optionally define required parameters
When training or predicting using your model, you may need the user to provide some inputs (eg number of epochs, batch size, test/train split, etc). To do this, you can create a [Pydantic BaseModel](https://pydantic.dev/docs/validation/latest/concepts/models/) which contains the parameters, their expected types, default values, and any additional validation rules. These parameters will then be requested from the user via a form in the UI when training / predicting.

### Step 3: Register the Model

The `@ModelRegistry.register()` decorator automatically registers your model with TokTagger. This requires the following parameters:

* name: The name to give your model (it will be called this within the UI)
* tasks: A list of tasks which this model is applicable for
* training_params: A Pydantic `BaseModel` containing required training parameters, optional
* prediction_params: A Pydantic `BaseModel` containing required prediction parameters, optional

#### Supported Tasks

Models can be registered for one or more of the following tasks:

- `"time-series"` - Time series annotation
- `"spectrogram"` - Spectrogram annotation
- `"video"` - Video frame annotation

### Step 4: Run Server with Custom Model

If you have added your own models, you must make sure they have been loaded before the server is run. You can run the server from within a Python script by initializing the `Server` class, and running `server.run()`:

```python
from toktagger.api.main import Server

# Import your custom model module so it gets registered
from my_models import MyCustomModel

# Create and run server
server = Server()
server.run()
```

## Data Handling

### Train/Validation/Test Split

Use the `split_data()` method provided by the base `Model` class to automatically split your data. If you wish to use this, you will need to include a parameter in your `BaseModel` similar to the following:

```py
class TrainingParameters(pydantic.BaseModel):
    train_val_test_split: list[typing.Annotated[float, pydantic.Field(gt=0, lt=1)]] = (
        pydantic.Field(
            min_length=3,
            max_length=3,
            description="Fraction of the total annotations to use in the training / validation / test sets. Fractions should sum to 1.",
        )
    )

    @pydantic.field_validator("train_val_test_split", mode="after")
    @classmethod
    def validate_sum_to_one(cls, value: list[float]) -> list[float]:
        if sum(value) != 1:
            raise ValueError("Train / Val / Test fractions must sum to 1!")
        return value
```

Then call the `split_data` method within your `train` method, using this parameter:

```python
def train(self, samples, annotations, params):
    # Split data into train/val/test sets
    self.split_data(
        samples=samples,
        annotations=annotations,
        train_val_test_split=params.train_val_test_split,
    )
    
    # Access the splits
    # self.train_samples, self.train_annotations
    # self.val_samples, self.val_annotations
    # self.test_samples, self.test_annotations
```

### Loading Sample Data

Access the project's data loader to retrieve actual data for each sample:

```python
from toktagger.api.core.data_loaders import LoaderRegistry
from toktagger.api.schemas.data import DataParams

# Initialize data loader
data_loader = LoaderRegistry.get(self.project.data_loader)()

# Get data for a sample
data = data_loader.get_sample(sample, DataParams())
```

## Progress Tracking

Use the `log_progress()` method to send training updates to the UI:

```python
# At the start of training
self.log_progress(training_status="started")

# During training (update progress and score)
self.log_progress(
    progress=int((epoch / num_epochs) * 100),
    score=validation_accuracy
)

# At the end of training
self.log_progress(
    training_status="completed",
    progress=100,
    score=final_accuracy
)
```

**Training Status Options:**
- `"queued"` - Model is waiting to start training
- `"started"` - Training has begun
- `"completed"` - Training finished successfully
- `"failed"` - Training encountered an error
- `"aborted"` - Training was manually stopped

## GPU Usage
TokTagger can reserve GPU nodes for ML Model tasks if requested by the user via the UI. If you wish to use a GPU to perform training or prediction in your model, then you can check whether a GPU is currently enabled on the model's worker node using this method:
```py
class MyModel(Model):
    def train(self, ...):

    if self.gpu_available():
        ...
```

Note that your model should be written in a way which makes it agnostic to the environment in which it is run. To do this, you may wish to include a `device` parameter within your Training and Prediction parameters, eg for a PyTorch model:
```py
class MyTrainParams(pydantic.BaseModel):
    device: typing.Literal["cpu", "cuda", "mps", "xpu"] = "cpu"
```
The exact format of the inputs required may depend on the ML framework you are using. You can then use these to correctly setup your model for the given environment, for example:
```py
class MyModel(Model):
    def train(
        self,
        samples: list[Sample],
        annotations: list[list[Annotation]],
        params: MyTrainParams,
        ):
        if not self.gpu_available() and params.device != "cpu":
            raise ValueError(
                "Only CPU available on current worker node, but non-CPU device requested!"
            )
        device = torch.device(params.device)
        self.model.to(device)
```
Note that a CPU is always available on all worker nodes, even if a GPU is requested. To reduce startup time, if a model exists on a worker node with a GPU but prediction is requested on a CPU node, that task will still be executed on the existing worker node which has GPU access. If a new model has a task scheduled which requires a GPU, and there are no free GPUs available, the least most recently used model will be stopped and its worker node will be made available to the new model.

!!! note
    Note that TokTagger uses Ray for task scheduling and management. Ray will detect Nvidia and most AMD GPUs reliably and make them available if requested, but for intel GPUs it may be less reliable. 
    
    For Macs with Apple silicon chips (such as the M1, M2, M3 series), the GPU is integrated and appears similarly to a CPU, and Ray will not detect it. Therefore it should be available to every worker node, regardless of the Allocate GPU setting, and regardless of what `self.gpu_available()` reports.

If Ray detects a GPU is available on your machine, it will automatically start TokTagger with GPU support enabled, and will use all GPU cores it is able to detect. If you wish to limit the number of GPUs which TokTagger should use in parallel, set the `MAX_GPU_ACTORS` environment variable. To disable GPUs, set `MAX_GPU_ACTORS` to 0.

!!! tip
    As mentioned above, Ray may fail to detect your GPU, especially on Mac devices. If you know how many GPU cores you can use in parallel on your machine, you can set `MAX_GPU_ACTORS` to that number. You should also set the environment variable `FORCE_NUM_GPUS` to `True`, as otherwise TokTagger will throw an error during start up as it tries to allocate more cores than it thinks are available.

    Note that this is just for task scheduling, and has no impact on the underlying resources available to each task. However the `Allocate GPU` button on the UI will now be enabled, and the `self.gpu_available()` will correctly report whether the user requested GPU usage or not for the given task, meaning that you can correctly assign tasks to GPU or CPU hardware accordingly.


## Complete Example: Random Forest Classification Model

Here's a complete example using a scikit-learn RandomForest for time series classification:

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import typing

from toktagger.api.models.base import Model, ModelRegistry
from toktagger.api.schemas.samples import Sample
from toktagger.api.schemas.annotations import Annotation, TimeInterval
from toktagger.api.schemas.data import TimeSeriesData, DataParams
from toktagger.api.core.data_loaders import LoaderRegistry

class TrainingParameters(pydantic.BaseModel):
    train_val_test_split: list[typing.Annotated[float, pydantic.Field(gt=0, lt=1)]] = (
        pydantic.Field(
            min_length=3,
            max_length=3,
            description="Fraction of the total annotations to use in the training / validation / test sets. Fractions should sum to 1.",
        )
    )

    @pydantic.field_validator("train_val_test_split", mode="after")
    @classmethod
    def validate_sum_to_one(cls, value: list[float]) -> list[float]:
        if sum(value) != 1:
            raise ValueError("Train / Val / Test fractions must sum to 1!")
        return value

@ModelRegistry.register("random_forest_classifier", ["time-series"], TrainingParameters)
class RandomForestModel(Model):
    """Random Forest classifier for time series classification"""
    
    def define_model(self):
        """Define a Random Forest classifier"""
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    
    def _extract_features(self, samples: list[Sample]) -> np.ndarray:
        """Extract statistical features from time series"""
        data_loader = LoaderRegistry.get(self.project.data_loader)()
        features = []
        
        for sample in samples:
            data: TimeSeriesData = data_loader.get_sample(
                sample,
                DataParams(),
            ).values["signal"]
            
            # Extract features: mean, std, min, max, etc.
            values = np.array(data.values)
            feature_vector = [
                np.mean(values),
                np.std(values),
                np.min(values),
                np.max(values),
                np.median(values),
                np.percentile(values, 25),
                np.percentile(values, 75),
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def _extract_labels(self, annotations: list[list[Annotation]]) -> np.ndarray:
        """Extract class labels from annotations"""
        labels = []
        for annotation_list in annotations:
            # Assume single classification annotation per sample
            label = annotation_list[0].label
            labels.append(label)
        return np.array(labels)
    
    def train(
        self,
        samples: list[Sample],
        annotations: list[list[Annotation]],
        params: TrainingParameters
    ) -> float:
        """Train the Random Forest model"""
        
        self.log_progress(training_status="started")
        
        # Split data
        self.split_data(
            samples=samples,
            annotations=annotations,
            train_val_test_split=params.train_val_test_split,
        )
        
        # Extract features and labels
        X_train = self._extract_features(self.train_samples)
        y_train = self._extract_labels(self.train_annotations)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Calculate accuracy
        train_accuracy = self.model.score(X_train, y_train) * 100
        
        # Evaluate on validation set if available
        if self.val_samples:
            X_val = self._extract_features(self.val_samples)
            y_val = self._extract_labels(self.val_annotations)
            val_accuracy = self.model.score(X_val, y_val) * 100
            final_accuracy = val_accuracy
        else:
            final_accuracy = train_accuracy
        
        self.log_progress(
            training_status="completed",
            progress=100,
            score=final_accuracy
        )
        
        return final_accuracy
    
    def predict(self, samples: list[Sample]) -> list[list[TimeInterval]]:
        """Make predictions on new samples"""
        
        # Extract features
        X = self._extract_features(samples)
        
        # Get predictions and probabilities
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Convert to annotations
        results = []
        for i, pred in enumerate(predictions):
            # Use (1 - max_probability) as uncertainty
            uncertainty = 1.0 - np.max(probabilities[i])
            
            # Get time range for the sample
            data_loader = LoaderRegistry.get(self.project.data_loader)()
            data = data_loader.get_sample(samples[i], DataParams())
            time_values = list(data.values.values())[0].time
            
            annotation = TimeInterval(
                validated=False,
                uncertainty=float(uncertainty),
                label=str(pred),
                start_time=time_values[0],
                end_time=time_values[-1],
                created_by=self.type,
            )
            results.append([annotation])
        
        return results
    
    def save(self, file_path: str):
        """Save model using joblib"""
        joblib.dump(self.model, file_path)
    
    def load(self, file_path: str):
        """Load model using joblib"""
        self.model = joblib.load(file_path)
```

### Running the Example

Create a script to load your model and start the server:

```python
# run_server.py
from toktagger.api.main import Server
from my_models import RandomForestModel  # Import to register

server = Server()
server.run()
```

Then run:
```sh
python run_server.py
```

## Using Docker

If you are using the docker compose option to run the server, you can provide a custom script to add your own models. Create a file that imports your models and runs the server:

```python
# custom_toktagger.py
from toktagger.api.main import Server
from my_models import MyCustomModel  # Import to register

server = Server()
server.run(
    host="0.0.0.0",
    port=8002
)
```

Then provide the path to your script when running docker compose:

```sh
CUSTOM_SCRIPT=./custom_toktagger.py docker compose --env-file .env.dev -f docker-compose.dev.yml up --build
```

!!! tip
    Make sure you provide the **path** to your script, not just the filename.
    
    e.g., provide `CUSTOM_SCRIPT=./custom_toktagger.py`, not `CUSTOM_SCRIPT=custom_toktagger.py`.

## Annotation Types

Different model types should return appropriate annotation types:

### `TimePoint`
For single time point predictions (e.g., disruption time):

```python
from toktagger.api.schemas.annotations import TimePoint

TimePoint(
    validated=False,
    uncertainty=0.1,
    label="Disruption",
    time=5.23,
    created_by="my_model",
)
```

### `TimeInterval`
For time range predictions (e.g., classification, mode detection):

```python
from toktagger.api.schemas.annotations import TimeInterval

TimeInterval(
    validated=False,
    uncertainty=0.05,
    label="H-mode",
    start_time=2.0,
    end_time=8.5,
    created_by="my_model",
)
```

## Best Practices

1. **Use `log_progress()` frequently** to keep the UI updated on training progress
2. **Implement early stopping** to avoid overfitting and save training time
3. **Return uncertainty estimates** to help annotators prioritize which predictions to review
4. **Validate on held-out data** to get realistic performance metrics
5. **Handle edge cases** like empty datasets or missing signals gracefully
6. **Document your model's assumptions** and requirements in the docstring
7. **Use appropriate data scaling** and normalization for your model type
8. **Save preprocessing parameters** along with model weights for consistent predictions

## Advanced: MC Dropout for Uncertainty

For neural networks, Monte Carlo (MC) Dropout is a powerful technique for uncertainty estimation:

```python
def predict_with_uncertainty(self, samples, num_mc_samples=20):
    """Use MC Dropout for uncertainty estimation"""
    
    # Keep dropout enabled
    self.model.train()
    
    all_predictions = []
    for _ in range(num_mc_samples):
        with torch.no_grad():
            predictions = self.model(data)
            all_predictions.append(predictions)
    
    # Stack and calculate statistics
    stacked = torch.stack(all_predictions)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0)
    
    return mean, std
```

This provides more reliable uncertainty estimates than single-pass predictions.
