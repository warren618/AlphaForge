"""Factor implementations directory.

Add your own factors here. Each factor file should use the
@register_factor decorator:

    from src.factor_pipeline.registry import register_factor

    @register_factor("my_momentum")
    def my_momentum(df: pd.DataFrame) -> pd.Series:
        return df["close"].pct_change(14)

The pipeline will auto-discover all registered factors.
See factors/example.py for a complete template.
"""
