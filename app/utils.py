def mapspeed_to_str(mapspeed):
    return {"-1": "HT", "0": "NM", "1": "DT", -1: "HT", 0: "NM", 1: "DT"}.get(
        mapspeed
    )


def mapspeed_to_int(mapspeed):
    return {"HT": -1, "NM": 0, "DT": 1}.get(mapspeed)
