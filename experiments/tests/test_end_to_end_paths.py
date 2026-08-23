import pandas as pd

from eurodreams.eurodreams_get_draws import parse_irish_archive
from euromillions import EuroMillionsGuess, evaluate_guess, normalize
from totoloto.totoloto_get_draws import Ranges, dedupe_sort, parse_year


def test_euromillions_pipeline_scores_ticket():
    csv_text = (
        "Date,Ball1,Ball2,Ball3,Ball4,Ball5,Lucky Star1,Lucky Star2\n"
        "2024-01-02,1,2,3,4,5,1,2\n"
        "2024-01-09,6,7,8,9,10,3,4\n"
    )
    df = normalize(csv_text)
    guess = EuroMillionsGuess([1, 2, 20, 21, 22], [1, 11])

    hits = [evaluate_guess(row, guess) for _, row in df.iterrows()]
    ball_avg = sum(h[0] for h in hits) / len(hits)
    star_avg = sum(h[1] for h in hits) / len(hits)

    assert ball_avg > 0
    assert star_avg > 0


def test_totoloto_end_to_end_metric():
    html = """
    <div>Sabado 17 de fevereiro de 2024</div>
    <ul><li>1</li><li>2</li><li>3</li><li>4</li><li>5</li><li>6</li></ul>
    """
    recs = dedupe_sort(parse_year(html, 2024, ranges=Ranges()))
    df = pd.DataFrame(recs)
    assert not df.empty

    guess = {1, 2, 10, 11, 12}
    bonus_guess = 6

    hits = []
    for _, row in df.iterrows():
        balls = {row[f"ball_{i}"] for i in range(1, 6)}
        bonus_hit = int(row["bonus"]) == bonus_guess
        hits.append((len(balls.intersection(guess)), bonus_hit))

    ball_mean = sum(h[0] for h in hits) / len(hits)
    bonus_rate = sum(1 for _, bonus in hits if bonus) / len(hits)

    assert ball_mean >= 1
    assert 0 <= bonus_rate <= 1


def test_eurodreams_end_to_end_metric():
    html = """
    <div>November 20th 2025</div>
    <ul><li>1</li><li>2</li><li>3</li><li>4</li><li>5</li><li>6</li><li>7</li></ul>
    """
    recs = parse_irish_archive(html)
    df = pd.DataFrame(recs)
    assert not df.empty

    guess = {2, 3, 4, 8, 9, 10}
    dream_guess = 7

    scored = []
    for _, row in df.iterrows():
        mains = {row[f"n{i}"] for i in range(1, 7)}
        scored.append((len(mains.intersection(guess)), row["dream"] == dream_guess))

    main_avg = sum(s[0] for s in scored) / len(scored)
    dream_rate = sum(1 for _, dream_hit in scored if dream_hit) / len(scored)

    assert main_avg > 0
    assert dream_rate in (0, 1)
