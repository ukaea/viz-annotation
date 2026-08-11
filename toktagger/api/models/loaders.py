import os
import ray
from ray.actor import ActorHandle
import pathlib
from toktagger.api.schemas.projects import Project
from toktagger.api.schemas.models import (
    Model,
    ModelUpdate,
    LoadParams,
    LocalLoadParams,
    GitlabLoadParams,
    HuggingfaceLoadParams,
)
from toktagger.api.core.sender import (
    send_model_updates,
)
import logging
import shutil
from mlflow import MlflowClient, MlflowException
from huggingface_hub import hf_hub_download
from safetensors import safe_open
import requests

logger = logging.getLogger("ray")
logger.setLevel("DEBUG")


class ModelLoader:
    def __init__(
        self,
        project: Project,
        model: Model,
        model_actor: ActorHandle,
        params: LoadParams,
    ):
        self.model = model
        self.model_actor = model_actor
        self.project = project
        self.params = params

        if not (models_dir := os.environ.get("MODEL_STORAGE")):
            raise ValueError("Model storage directory not provided to worker node.")

        self.models_dir = pathlib.Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        send_model_updates(
            project_id=project.id,
            model_id=model.id,
            updates=ModelUpdate(training_status="started"),
        )

    def _attempt_load(
        self, weights_path: pathlib.Path, remove_if_failed: bool = False
    ) -> dict[str, str]:
        results_dir = self.models_dir.joinpath(self.model.id)

        # Try loading actor with weights file, catch and reraise any errors
        try:
            load_temp_weights_task = self.model_actor.wrapped_load.remote(
                results_dir=weights_path.parent, weights_filename=weights_path.name
            )
            ray.get(load_temp_weights_task)

            # Save the model with the correct dir path
            results_dir.mkdir()
            save_weights_task = self.model_actor.wrapped_save.remote(results_dir)
            ray.get(save_weights_task)

        except Exception as e:
            if remove_if_failed:
                weights_path.unlink()
            # Also delete directory of results, if it has already been created
            if results_dir.exists():
                shutil.rmtree(results_dir)
            return self._log_error(e)

        send_model_updates(
            project_id=self.project.id,
            model_id=self.model.id,
            updates=ModelUpdate(training_status="completed", progress=100),
        )

        return {
            "project_id": self.project.id,
            "model_id": self.model.id,
            "message": None,
        }

    def _log_error(
        self, error: Exception, message: str | None = None
    ) -> dict[str, str]:
        logger.error(error)

        if not message:
            err_lines = str(error).strip().splitlines()
            message = err_lines[-1] if err_lines else repr(error)

        send_model_updates(
            project_id=self.project.id,
            model_id=self.model.id,
            updates=ModelUpdate(training_status="failed"),
        )

        return {
            "project_id": self.project.id,
            "model_id": self.model.id,
            "message": message,
        }

    def _check_safetensor(
        self, weights_path: str | pathlib.Path
    ) -> dict[str, str] | None:
        if os.environ.get("MODELS_SAFETENSORS_ONLY", "False").lower() == "true":
            try:
                with safe_open(weights_path, framework="pt", device="cpu") as f:
                    _ = list(f.keys())
            except Exception as e:
                # Not a safetensors object, return error
                return self._log_error(e, message="retrieved file is not a SafeTensor!")
            return

    def load(self) -> dict[str, str]:
        pass


class LocalLoader(ModelLoader):
    def __init__(
        self,
        project: Project,
        model: Model,
        model_actor: ActorHandle,
        params: LocalLoadParams,
    ):
        if not isinstance(params, LocalLoadParams):
            raise TypeError(
                f"Expected loader params of type 'LocalLoadParams' but got '{type(params)}'"
            )
        super().__init__(
            model=model, model_actor=model_actor, project=project, params=params
        )

    def load(self) -> dict[str, str]:
        # Check worker can see weights file
        weights_path = pathlib.Path(self.params.weights_path)
        if not weights_path.exists():
            return self._log_error(
                error=FileNotFoundError(
                    f"Worker node cannot find file at {weights_path}"
                )
            )

        if unsafe := self._check_safetensor(weights_path=weights_path):
            return unsafe

        return self._attempt_load(weights_path=weights_path, remove_if_failed=False)


class GitlabLoader(ModelLoader):
    def __init__(
        self,
        project: Project,
        model: Model,
        model_actor: ActorHandle,
        params: GitlabLoadParams,
    ):
        if not isinstance(params, GitlabLoadParams):
            raise TypeError(
                f"Expected loader params of type 'GitlabLoadParams' but got '{type(params)}'"
            )
        super().__init__(
            model=model, model_actor=model_actor, project=project, params=params
        )

    def _get_model_version(self):
        # Pull object from ML Model registry
        client = MlflowClient()
        if self.params.model_version:
            mlflow_model = client.get_model_version(
                self.params.model_name, self.params.model_version
            )
        else:
            mlflow_model = max(
                client.get_latest_versions(self.params.model_name),
                key=lambda mv: int(mv.version),
                default=None,
            )

        if not mlflow_model:
            raise MlflowException("Failed to access model versions from Gitlab.")

        return mlflow_model

    def _download_artifact(self, model_version: int) -> pathlib.Path:
        # Note that it seems like download_artifacts and list_artifacts methods are broken
        # https://gitlab.com/gitlab-org/gitlab/-/work_items/591960
        # Will perform a workaround by downloading directly from API
        download_path = self.models_dir.joinpath(
            pathlib.Path(self.params.weights_path).name
        )
        with requests.get(
            f"{os.environ.get('MODELS_GITLAB_URL')}/api/v4/projects/{self.params.gitlab_project_id}/packages/ml_models/{model_version}/files/{self.params.weights_path}",
            headers={"Authorization": f"Bearer {os.environ['MLFLOW_TRACKING_TOKEN']}"},
            stream=True,
            timeout=600,
        ) as response:
            response.raise_for_status()
            with download_path.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)

        return download_path

    def load(self) -> dict[str, str]:
        # Construct URI required
        if not all(
            (
                os.environ.get("MODELS_GITLAB_URL"),
                os.environ.get("MODELS_GITLAB_TOKEN"),
                self.params.gitlab_project_id,
            )
        ):
            return self._log_error(
                ValueError("Missing required parameters"),
                "Gitlab URL, Token or Project ID not specified when trying to load ML model!",
            )

        # Set env vars needed for Mlflow client to connect to Gitlab
        os.environ["MLFLOW_TRACKING_URI"] = (
            f"{os.environ.get('MODELS_GITLAB_URL')}/api/v4/projects/{self.params.gitlab_project_id}/ml/mlflow"
        )
        os.environ["MLFLOW_TRACKING_TOKEN"] = os.environ.get("MODELS_GITLAB_TOKEN")

        try:
            mlflow_version = self._get_model_version()
        except MlflowException as e:
            return self._log_error(
                error=e,
                message="requested version of selected model could not be found!",
            )

        try:
            download_path = self._download_artifact(mlflow_version.version)
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            if status_code == 404:
                err_msg = "could not find model weights at provided file location!"
            elif status_code == 403:
                err_msg = "Gitlab token does not have the correct permissions to access this model!"
            else:
                err_msg = "server error from Gitlab!"

            return self._log_error(error=e, message=err_msg)

        except requests.exceptions.Timeout as e:
            download_path.unlink(missing_ok=True)
            return self._log_error(error=e, message="download timed out!")

        if unsafe := self._check_safetensor(download_path):
            # Delete downloaded file
            download_path.unlink()
            return unsafe

        result = self._attempt_load(weights_path=download_path, remove_if_failed=False)
        download_path.unlink()
        return result


class HuggingfaceLoader(ModelLoader):
    def __init__(
        self,
        project: Project,
        model: Model,
        model_actor: ActorHandle,
        params: HuggingfaceLoadParams,
    ):
        if not isinstance(params, HuggingfaceLoadParams):
            raise TypeError(
                f"Expected loader params of type 'HuggingfaceLoadParams' but got '{type(params)}'"
            )
        super().__init__(
            model=model, model_actor=model_actor, project=project, params=params
        )

    def load(self) -> dict[str, str]:
        # Pull object from Hugging Face
        try:
            weights_path = hf_hub_download(
                repo_id=f"{self.params.huggingface_userspace}/{self.params.model_name}",
                filename=self.params.weights_path,
                revision=self.params.model_version,
                local_dir=self.models_dir,
            )
        except Exception as e:
            return self._log_error(
                error=e, message="requested model could not be found!"
            )

        download_path = pathlib.Path(weights_path)
        if unsafe := self._check_safetensor(download_path):
            # Delete downloaded file
            download_path.unlink()
            return unsafe

        result = self._attempt_load(weights_path=download_path, remove_if_failed=False)
        download_path.unlink()
        return result
