INSTRUCTIONS = """
    TokTagger is a human-in-the-loop annotation platform for fusion tokamak data.

    Important:
    - Use TokTagger MCP tools rather than attempting to call REST API endpoints directly.
    - Do not invent project_id or sample_id values.
    - Discover project_ids with toktagger_get_projects, searching by project name if known.
    - Discover sample_ids with toktagger_get_samples, searching by shot ID if known.
    
    Users organise data into Projects, which defines the labelling task and data loading method to use.
    Each project can have one of the following Tasks: time-series, profile-2d, or video.
    There are some built in data loaders, including local files, SAL, and UDA, or users can define their own.
    To find the data loaders available, use toktagger_get_dataloaders.
    
    Within Projects, users can create Samples which each correspond to a single experimental shot (or pulse).
    These will have a human readable shot_id, not to be confused with the sample_id which is used for access within TokTagger.
    They will contain data which is either time-series (single or multivariate), profile-2D (spectrograms) or video data encoded as base64 strings.
    
    Users then create Annotations within each Sample. These designate areas of interest which ML models may want to be trained on later.
    These can either be generated manually by the user (via the UI), by automated annotators (instant), or by ML models (scheduled asynchronous tasks).
    Annotations can have different forms depending on the data being annotated.
    
    Once a human has annotated and validated annotations for a set of samples, ML models can be trained on the data from within TokTagger.
    Alternatively, model weights can be loaded either via local file, Gitlab, or Hugging Face.
    Predictions can then be made, either for a set of Samples, or for an individual Sample.
    Humans will then inspect the predictions, correct them if necessary, and save them as validated to increase the training dataset.
    
    Recommended workflow:

    1. Call toktagger_get_projects to discover available projects, their project_ids and metadata.
    3. Call toktagger_get_samples to browse project samples metadata and their sample_ids, providing the project_id from above.
    5. Call toktagger_get_sample_data_summary to retrieve information about diagnostic data for the sample, providing the project_id and sample_id from above
    6. Call toktagger_get_sample_annotations to inspect existing annotations, providing the project_id and sample_id from above
    7. Call toktagger_get_annotator_types to find available annotators for the task associated with this project
    8. Call toktagger_create_automated_sample_annotations to create annotations with the selected annotator, prompting the user for required input parameters
    9. Call toktagger_update_sample_annotations to update annotations from either human or automated sources, providing the project_id and sample_id from above 

    Model workflow:

    1. Call toktagger_get_model_types to discover available model types for a project's task.
    2. Call toktagger_get_model_training_schema to inspect training parameters, and prompt them to the user.
    3. Call toktagger_start_model_training to train a model for a given project, with user provided parameters.
    4. Call toktagger_get_model_training_info to monitor progress.
    5. Call toktagger_get_model_prediction_schema to inspect prediction parameters, and prompt them to the user.
    6. Call toktagger_create_model_predictions / toktagger_create_sample_model_predictions to generate predictions.
    
    Asynchronous operations:

    The following operations will return a task_id rather than an immediate result:
    - toktagger_start_model_training
    - toktagger_load_model_weights_local
    - toktagger_load_model_weights_gitlab
    - toktagger_load_model_weights_hugging_face
    - toktagger_create_model_predictions
    - toktagger_create_sample_model_predictions

    Use the corresponding status endpoints to monitor progress for the returned task_id.

    Notes:
    
    - Automated annotators and ML models can only operate on projects with relevant Tasks - find the task from toktagger_get_projects, 
      then check available annotators with toktagger_get_annotator_types, and/or ML models with toktagger_get_model_types

    """
