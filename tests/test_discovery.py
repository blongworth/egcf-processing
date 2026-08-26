from egcf_processing.discovery import find_all_files, find_gems_files, find_surface_files


def test_find_surface_files_sorts_by_rotation_ts_and_skips_zero_byte(tmp_path):
    later = tmp_path / "surface_2026-08-25-18-32_lander.log"
    earlier = tmp_path / "surface_2026-08-25-14-53_lander.log"
    empty = tmp_path / "surface_2026-08-25-20-00_lander.log"
    later.write_text("V:2026-08-25T18:32:00Z,C1,Re\n")
    earlier.write_text("V:2026-08-25T14:53:00Z,C1,Re\n")
    empty.write_text("")

    files = find_surface_files(tmp_path)
    assert files == [earlier, later]


def test_find_surface_files_excludes_events_log(tmp_path):
    lander = tmp_path / "surface_2026-08-25-18-32_lander.log"
    events = tmp_path / "surface_2026-08-25-18-32_events.log"
    lander.write_text("V:2026-08-25T18:32:00Z,C1,Re\n")
    events.write_text("2026-08-25T18:33:00Z SYSTEM startup complete\n")

    assert find_surface_files(tmp_path) == [lander]


def test_find_all_files_merges_gems_and_surface_in_rotation_order(tmp_path):
    gems = tmp_path / "gems_2026-08-25-08-00.txt"
    surface = tmp_path / "surface_2026-08-25-14-53_lander.log"
    gems.write_text("V:2026-08-25T08:00:00Z,C1,Re\n")
    surface.write_text("V:2026-08-25T14:53:00Z,C1,Re\n")

    assert find_all_files(tmp_path) == [gems, surface]
    assert find_gems_files(tmp_path) == [gems]
