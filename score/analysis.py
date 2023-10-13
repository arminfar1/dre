import polars as pl
import pyarrow


def get_prediction_summary_pl(pred_data):
    # Collect the prediction data table to Polars via Arrow.
    pred_data_pl = pl.from_arrow(pyarrow.Table.from_batches(pred_data._collect_as_arrow()))
    pred_col_names = [
        "liftWb",
        "liftWbStandardError",
        "liftMlaGnuA",
        "liftMlaGnuB",
        "pDenomMlaGnuA",
        "pDenomMlaGnuB",
        "pNumMlaGnuA",
        "pNumMlaGnuB",
    ]
    # Calculate the aggregate summary of prediction columns
    pred_summary = (
        pred_data_pl.melt(id_vars="conversionType", value_vars=pred_col_names)
        .with_columns(
            [(pl.col("value") <= 0).alias("neg"), (pl.col("value") >= 1).alias("onepluse")]
        )
        .groupby("variable")
        .agg(
            [
                pl.min("value").alias("min"),
                pl.max("value").alias("max"),
                pl.mean("value").alias("mean"),
                pl.std("value").alias("sd"),
                ((pl.col("value") - pl.col("value").mean()) ** 3).mean().alias("skewness"),
                pl.quantile("value", 0.5).alias("p50"),
                pl.quantile("value", 0.25).alias("p25"),
                pl.quantile("value", 0.1).alias("p10"),
                pl.quantile("value", 0.75).alias("p75"),
                pl.quantile("value", 0.9).alias("p90"),
                pl.mean("neg").alias("negative_frac"),
                pl.mean("onepluse").alias("oneplus_frac"),
            ]
        )
        .with_columns((pl.col("skewness") / (pl.col("sd") ** 6)).alias("skewness"))
        .transpose(include_header=True, column_names=pred_col_names)
    )
    return pred_summary
