import pyethnicity

ZCTA = 11106
TRACT = 72153750502
RACES = ["asian", "black", "hispanic", "white"]


def arr_equal(arr1, arr2):
    for i, j in zip(arr1, arr2):
        if i != j:
            return False

    return True


def test_bisg():
    df = pyethnicity.bisg("luo", TRACT, "tract")
    assert arr_equal(df.columns, ["last_name", "tract"] + RACES)

    df = pyethnicity.bisg("luo", ZCTA, "zcta")
    assert arr_equal(df.columns, ["last_name", "zcta"] + RACES)

    df = df.drop("last_name", axis=1)
    for other in ["Luo", "lUo", "luo jr."]:
        assert df.equals(
            pyethnicity.bisg(other, ZCTA, "zcta").drop("last_name", axis=1)
        )


def test_bifsg():
    df = pyethnicity.bifsg("mercy", "luo", TRACT, "tract")
    assert arr_equal(df.columns, ["first_name", "last_name", "tract"] + RACES)

    df = pyethnicity.bifsg("mercy", "luo", ZCTA, "zcta")
    assert arr_equal(df.columns, ["first_name", "last_name", "zcta"] + RACES)

    df = df.drop(["first_name", "last_name"], axis=1)
    for other_fn, other_ln in zip(
        ["MErcY", "mercy12", "mercy sr."], ["Luo", "lUo", "luo jr."]
    ):
        assert df.equals(
            pyethnicity.bifsg(other_fn, other_ln, ZCTA, "zcta").drop(
                ["first_name", "last_name"], axis=1
            )
        )
