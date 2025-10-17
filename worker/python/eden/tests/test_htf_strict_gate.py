import pandas as pd
from eden.features.htf_ict_bias import build_htf_context


def test_build_htf_context_columns():
    idx = pd.date_range('2024-01-01', periods=5, freq='H', tz='UTC')
    df1 = pd.DataFrame({'open':[1,2,3,4,5],'high':[2,3,4,5,6],'low':[0,1,2,3,4],'close':[1.5,2.5,3.5,4.5,5.5],'volume':[100]*5}, index=idx)
    df4 = df1.copy()
    ctx = build_htf_context(df1, df4)
    for col in ['HTF_FVG_BULL','HTF_FVG_BEAR','HTF_OB_COUNT_BULL','HTF_OB_COUNT_BEAR','HTF_RECENT_SWEEP_HIGH','HTF_RECENT_SWEEP_LOW']:
        assert col in ctx.columns