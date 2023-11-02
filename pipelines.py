from .inputs.transformers import NoMarketingMaskTransformer
from pyspark.ml import Pipeline


class ScoringPipeline:
    def __init__(self, s3_output_path):
        self.scoring_pipeline_model = None
        self.s3_output_path = s3_output_path

    def create_scoring_pipeline(
        self, featurization_pipeline_model, sampled_process_data, marketing_index=None, model=None
    ):
        nm_mask = marketing_index
        nm_mask_transformer = NoMarketingMaskTransformer(nm_mask)
        if model:
            stages = featurization_pipeline_model.getStages() + [nm_mask_transformer, model]
        else:
            stages = featurization_pipeline_model.getStages()
        scoring_pipeline = Pipeline(stages=stages)
        self.scoring_pipeline_model = scoring_pipeline.fit(sampled_process_data)

    def persist_scoring_pipeline(self):
        self.scoring_pipeline_model.write().overwrite().save(
            f"{self.s3_output_path}/data_pipeline/"
        )

    def get_scoring_pipeline(self):
        return self.scoring_pipeline_model
