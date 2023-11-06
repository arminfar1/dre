from .inputs.transformers import NoMarketingMaskTransformer
from pyspark.ml import Pipeline


class ScoringPipeline:
    """
    The class to create and manage a scoring pipeline. This pipeline prepares the data for scoring.
    """

    def __init__(self, s3_output_path):
        """
        Initializes the ScoringPipeline with a specified output path for saving the scoring pipeline model.

        Parameters:
            s3_output_path (str): The S3 path where the scoring pipeline model will be saved.
        """
        self.scoring_pipeline_model = None
        self.s3_output_path = s3_output_path

    def create_scoring_pipeline(
        self, featurization_pipeline_model, sampled_process_data, marketing_index=None, model=None
    ):
        """
        Creates a scoring pipeline by optionally adding a NoMarketingMaskTransformer to
        the featurization pipeline model.

        Parameters:
            featurization_pipeline_model (PipelineModel): The pre-built featurization pipeline model.
            sampled_process_data (DataFrame): The data to be used to fit the scoring pipeline.
            marketing_index (int, optional): The index of the marketing feature to be masked. Defaults to None.
            model (Model, optional): The model to be included in the scoring pipeline. Defaults to None. This will
            be required only for OLS model.
        """
        nm_mask = marketing_index
        nm_mask_transformer = NoMarketingMaskTransformer(nm_mask)
        if model:
            stages = featurization_pipeline_model.getStages() + [nm_mask_transformer, model]
        else:
            stages = featurization_pipeline_model.getStages()
        scoring_pipeline = Pipeline(stages=stages)
        self.scoring_pipeline_model = scoring_pipeline.fit(sampled_process_data)

    def persist_scoring_pipeline(self):
        """
        Saves the scoring pipeline model to the specified S3 output path.
        """
        self.scoring_pipeline_model.write().overwrite().save(
            f"{self.s3_output_path}/data_pipeline/"
        )

    def get_scoring_pipeline(self):
        return self.scoring_pipeline_model
