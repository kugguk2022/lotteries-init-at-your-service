from euromillions.get_draws import normalize


def test_normalize_parses_minimal_csv():
    csv_text = (
        "Date,Ball1,Ball2,Ball3,Ball4,Ball5,Lucky Star1,Lucky Star2\n"
        "2024-01-02,1,2,3,4,5,1,2\n"
        "2024-01-09,6,7,8,9,10,3,4\n"
    )

    df = normalize(csv_text)

    assert len(df) == 2
    assert df.iloc[0]["ball_1"] == 1
    assert df.iloc[1]["star_2"] == 4


def test_normalize_dedupes_and_handles_whitespace():
    csv_text = (
        " Date ,Ball 1 ,Ball 2 ,Ball 3 ,Ball 4 ,Ball 5 ,Lucky Star 1 ,Lucky Star 2 ,Ignore\n"
        "2024-01-09 ,6,7,8,9,10,3,4,foo\n"
        "2024-01-02 ,1,2,3,4,5,1,2,bar\n"
        "2024-01-09 ,6,7,8,9,10,3,4,baz\n"
    )

    df = normalize(csv_text)

    assert list(df.draw_date.astype(str)) == ["2024-01-02", "2024-01-09"]
    assert df.iloc[-1]["ball_5"] == 10


def test_normalize_handles_national_lottery_format():
    csv_text = (
        "DrawDate,Ball 1,Ball 2,Ball 3,Ball 4,Ball 5,Lucky Star 1,Lucky Star 2,UK Millionaire Maker,DrawNumber\n"
        '25-Nov-2025,6,11,17,35,44,3,7,"JWGH03530",1897\n'
        '21-Nov-2025,17,19,29,35,48,5,9,"HVFV75870",1896\n'
    )

    df = normalize(csv_text)

    assert list(df.draw_date.astype(str)) == ["2025-11-21", "2025-11-25"]
    assert df.iloc[-1]["ball_1"] == 6
    assert df.iloc[0]["star_2"] == 9
