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
