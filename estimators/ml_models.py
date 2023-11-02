from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.mllib.evaluation import MulticlassMetrics


class SparkBinaryClassificationMetrics:
    def __init__(self, dataset, labelCol, predictionCol, probabilityCol, rawPredictionCol):
        self.bc_valuator = None
        self.mc_valuator = None
        self.dataset = dataset
        self.label_col = labelCol
        self.prediction_col = predictionCol
        self.probability_col = probabilityCol
        self.raw_prediction = rawPredictionCol
        self.metrics_dict = dict()
        self.build_metrics_objects()

    def build_metrics_objects(self):
        # Instantiate evaluators
        self.bc_valuator = BinaryClassificationEvaluator(labelCol=self.label_col)
        self.bc_valuator.setRawPredictionCol(self.raw_prediction)

        self.mc_valuator = MulticlassClassificationEvaluator(labelCol=self.label_col)
        self.mc_valuator.setPredictionCol(self.prediction_col)
        self.mc_valuator.setProbabilityCol(self.probability_col)

    def get_metrics(self):
        metric_labels = ["area_roc", "area_prc", "accuracy", "f1", "precision", "recall"]
        metricKeys = [f"{ml}" for ml in metric_labels]
        # Capture metrics -> areas, acc, f1, prec, rec
        area_roc = round(
            self.bc_valuator.evaluate(self.dataset, {self.bc_valuator.metricName: "areaUnderROC"}),
            5,
        )
        area_prc = round(
            self.bc_valuator.evaluate(self.dataset, {self.bc_valuator.metricName: "areaUnderPR"}), 5
        )
        acc = round(
            self.mc_valuator.evaluate(self.dataset, {self.mc_valuator.metricName: "accuracy"}), 5
        )
        f1 = round(self.mc_valuator.evaluate(self.dataset, {self.mc_valuator.metricName: "f1"}), 5)
        prec = round(
            self.mc_valuator.evaluate(
                self.dataset, {self.mc_valuator.metricName: "weightedPrecision"}
            ),
            5,
        )
        rec = round(
            self.mc_valuator.evaluate(
                self.dataset, {self.mc_valuator.metricName: "weightedRecall"}
            ),
            5,
        )

        # Create a metric values array
        metric_values_array = []
        metric_values_array.extend((area_roc, area_prc, acc, f1, prec, rec))

        # Zip the keys and values into a dictionary
        metrics_dictionary = dict(zip(metricKeys, metric_values_array))

        return metrics_dictionary


class SparkMultiClassClassificationMetrics:
    def __init__(
        self, dataset, labelCol, labels_list, predictionCol, probabilityCol, rawPredictionCol
    ):
        self.bc_valuator = None
        self.mc_valuator = None
        self.dataset = dataset
        self.label_col = labelCol
        self.prediction_col = predictionCol
        self.probability_col = probabilityCol
        self.raw_prediction = rawPredictionCol
        self.metrics_dict = dict()
        self.build_metrics_objects()
        self.labels: [] = labels_list

    def build_metrics_objects(self):
        self.mc_valuator = MulticlassClassificationEvaluator(labelCol=self.label_col)
        self.mc_valuator.setPredictionCol(self.prediction_col)
        self.mc_valuator.setProbabilityCol(self.probability_col)

    @property
    def get_evaluator(self):
        return self.mc_valuator

    def get_metrics(self):
        metric_labels = ["accuracy", "f1", "precision", "recall"]
        metricKeys = [f"{ml}" for ml in metric_labels]

        acc = round(
            self.mc_valuator.evaluate(self.dataset, {self.mc_valuator.metricName: "accuracy"}), 5
        )
        f1 = round(self.mc_valuator.evaluate(self.dataset, {self.mc_valuator.metricName: "f1"}), 5)
        prec = round(
            self.mc_valuator.evaluate(
                self.dataset, {self.mc_valuator.metricName: "weightedPrecision"}
            ),
            5,
        )
        rec = round(
            self.mc_valuator.evaluate(
                self.dataset, {self.mc_valuator.metricName: "weightedRecall"}
            ),
            5,
        )

        metric_values_array = [acc, f1, prec, rec]

        metrics_dictionary = dict(zip(metricKeys, metric_values_array))

        return metrics_dictionary

    def get_class_metrics(self):
        # Convert the dataset to RDD of (prediction, label) pairs
        predictionAndLabels = self.dataset.select(self.prediction_col, self.label_col).rdd.map(
            tuple
        )

        # Instantiate the MulticlassMetrics object
        metrics = MulticlassMetrics(predictionAndLabels)
        weighted_metrics = {
            "weighted_recall": metrics.weightedRecall,
            "weighted_precision": metrics.weightedPrecision,
            "Weighted F(1) Score": metrics.weightedFMeasure(),
        }
        class_metrics = {}
        for index, label in enumerate(self.labels):
            class_metrics[f"class_{label}"] = {
                "precision": metrics.precision(float(index)),
                "recall": metrics.recall(float(index)),
                "f1-score": metrics.fMeasure(float(index)),
            }

        return class_metrics, weighted_metrics


class CrossValidation:
    def __init__(
        self, dataset, model, label_col, predictionCol, probabilityCol, rawPredictionCol, numFolds=5
    ):
        self.dataset = dataset
        self.model = model
        self.label_col = label_col
        self.prediction_col = predictionCol
        self.probability_col = probabilityCol
        self.raw_prediction = rawPredictionCol
        self.numFolds = numFolds

    def run(
        self,
    ):
        paramGrid = (
            ParamGridBuilder()
            .addGrid(self.model.maxDepth, [10, 15, 20, 30, 50])
            .addGrid(self.model.numTrees, [10, 30, 50])
            .addGrid(self.model.maxBins, [32, 64, 128])
            .addGrid(self.model.impurity, ["gini", "entropy"])
            .addGrid(self.model.subsamplingRate, [0.7, 0.8, 0.9, 1.0])
            .addGrid(self.model.minInfoGain, [0.0, 0.1, 0.2, 0.3, 0.5])
            .addGrid(self.model.minInstancesPerNode, [1, 5, 10, 15])
            .addGrid(self.model.featureSubsetStrategy, ["sqrt", "log2", "onethird"])
            .build()
        )

        evaluator = MulticlassClassificationEvaluator(
            labelCol=self.label_col,
            predictionCol=self.prediction_col,
            probabilityCol=self.probability_col,
        )

        # Create the CrossValidator
        cv = CrossValidator(
            estimator=self.model,
            estimatorParamMaps=paramGrid,
            evaluator=evaluator,
            numFolds=self.numFolds,
        )

        # Fit the model using cross-validation
        cvModel = cv.fit(self.dataset)
        bestModel = cvModel.bestModel

        # print the best hyperparam
        best_parms = bestModel.extractParamMap()

        return bestModel, best_parms
