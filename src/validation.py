import pandera as pa
from pandera import Column, DataFrameSchema, Check


schema = DataFrameSchema(
    {
        "Quantity": Column(int, Check.in_range(1, 5)),
        "UnitPrice": Column(float, Check.greater_than(0)),
        "TotalPrice": Column(float, Check.greater_than(0)),
        "ItemsInCart": Column(int, Check.in_range(1, 10)),
        "RevenuePerItem": Column(float, Check.greater_than(0)),
        "CartUtilizationRate": Column(float, Check.in_range(0.0, 1.0)),
        "IsHighValueOrder": Column(int, Check.isin([0, 1])),
        "HasCoupon": Column(int, Check.isin([0, 1])),
        "OrderMonth": Column(int, Check.in_range(1, 12)),
    },
    coerce=False,
)


def validate(df):
    """
    Runs the schema check with lazy=True so it collects every single
    violation before raising — instead of stopping at the first one.
    Much easier to debug when you can see all the problems at once.

    Returns the validated dataframe if everything passes.
    """
    try:
        validated = schema.validate(df, lazy=True)
        print(f"Validation passed. Shape: {validated.shape}")
        return validated
    except pa.errors.SchemaErrors as e:
        print("Validation failed. Issues found:")
        print(e.failure_cases.to_string())
        raise