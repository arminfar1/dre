from .inputs.transformers import NoMarketingMaskTransformer
from pyspark.ml import Pipeline


class ScoringPipeline:
    def __init__(self, s3_output_path):
        self.scoring_pipeline = None
        self.s3_output_path = s3_output_path

    def create_scoring_pipeline(self, featurization_pipeline, marketing_index=None, model=None):
        nm_mask = marketing_index
        nm_mask_transformer = NoMarketingMaskTransformer(nm_mask)
        if model:
            scoring_pipeline = Pipeline(
                stages=featurization_pipeline.getStages() + [nm_mask_transformer, model]
            )
        else:
            scoring_pipeline = Pipeline(stages=featurization_pipeline.getStages())
        self.scoring_pipeline = scoring_pipeline

    def persist_scoring_pipeline(self):
        self.scoring_pipeline.write().overwrite().save(f"{self.s3_output_path}/data_pipeline/")

    def get_scoring_pipeline(self):
        return self.scoring_pipeline
