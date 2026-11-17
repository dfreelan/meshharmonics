import json

import pytest

from meshharmonics.cli import main


@pytest.fixture
def sphere_file(tmp_path):
    path = tmp_path / "s.obj"
    assert main(["primitive", "icosphere", str(path), "--subdivisions", "1"]) == 0
    return path


def test_info_json(sphere_file, capsys):
    capsys.readouterr()
    assert main(["info", str(sphere_file), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["n_vertices"] == 42 and report["is_watertight"] and report["genus"] == 0
    assert report["valence"]["histogram"] == {"5": 12, "6": 30}


def test_info_text_on_fixture(data_dir, capsys):
    assert main(["info", str(data_dir / "fin.off")]) == 0
    out = capsys.readouterr().out
    assert "non-manifold edges  1" in out and "manifold            no" in out


def test_spectrum_and_hks(sphere_file, tmp_path, capsys):
    assert main(["spectrum", str(sphere_file), "-k", "4", "--json", "--shape-dna"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["eigenvalues"]) == 4 and abs(payload["eigenvalues"][0]) < 1e-8
    out = tmp_path / "hks.csv"
    assert main(["hks", str(sphere_file), "-k", "8", "--times", "3", "-o", str(out)]) == 0
    lines = out.read_text().splitlines()
    assert len(lines) == 43 and lines[0].startswith("vertex,t=")


def test_geodesic_convert_and_sh(sphere_file, tmp_path, capsys):
    assert main(["geodesic", str(sphere_file), "-s", "0", "--json"]) == 0
    dist = json.loads(capsys.readouterr().out)["distances"]
    assert dist[0] == 0.0 and max(dist) > 2.5
    assert main(["geodesic", str(sphere_file), "-s", "0", "-s", "1", "--method", "dijkstra"]) == 0
    off = tmp_path / "s.off"
    assert main(["convert", str(sphere_file), str(off), "--clean", "--orient"]) == 0
    assert off.read_text().startswith("OFF")
    capsys.readouterr()
    assert main(["sh", str(off), "--lmax", "3", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["band_norms"][0] == 1.0


def test_errors_return_status_2(tmp_path, capsys):
    bad = tmp_path / "bad.obj"
    bad.write_text("v 0 0 0\nf 1 2 3\n")
    assert main(["info", str(bad)]) == 2
    assert "bad.obj:2" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["frobnicate"])
