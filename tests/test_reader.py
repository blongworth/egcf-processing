from egcf_processing.reader import read_all, read_file


def test_read_file_skips_malformed_lines(tmp_path):
    path = tmp_path / "gems_2026-06-02-14-00.txt"
    path.write_text(
        "\n".join(
            [
                "R:2026-06-02T14:00:00Z,2,100",
                "garbage line",
                "R:2026-06-02T14:00:01Z,15,200",
                "",
            ]
        )
    )
    records = read_file(path)
    assert len(records) == 2
    assert all(r["source_file"] == path.name for r in records)


def test_read_all_concatenates_in_order(tmp_path):
    path_a = tmp_path / "gems_2026-06-02-14-00.txt"
    path_b = tmp_path / "gems_2026-06-02-18-00.txt"
    path_a.write_text("R:2026-06-02T14:00:00Z,2,100\n")
    path_b.write_text("R:2026-06-02T18:00:00Z,2,100\n")

    records = read_all([path_a, path_b])
    assert [r["source_file"] for r in records] == [path_a.name, path_b.name]


def test_read_surface_file_uses_embedded_lander_ts_not_surface_receipt_ts(tmp_path):
    path = tmp_path / "surface_2026-08-25-14-53_lander.log"
    path.write_text(
        "\n".join(
            [
                "# surface-lander-log-v1",
                "# fields: iso8601 payload",
                "2026-08-25T14:53:46Z V:2026-08-25T14:53:45Z,C1,Fl",
                "2026-08-25T14:53:16Z S,Off,SPD=1200,TURBO=not ready,RGA=off",
                "",
            ]
        )
    )
    records = read_file(path)
    assert len(records) == 1
    assert records[0]["tag"] == "V"
    assert records[0]["ts"].isoformat() == "2026-08-25T14:53:45"
    assert records[0]["source_file"] == path.name


def test_read_surface_file_skips_lines_without_a_payload(tmp_path):
    path = tmp_path / "surface_2026-08-25-14-53_lander.log"
    path.write_text("2026-08-25T14:53:46Z\n")

    assert read_file(path) == []
