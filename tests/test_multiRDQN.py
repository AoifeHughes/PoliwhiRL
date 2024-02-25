import pytest
from main import main
import shutil
import os
from unittest.mock import patch, MagicMock

@pytest.fixture
def clone_current_directory_to_temp(tmp_path):
    current_dir = os.getcwd()
    temp_dir = tmp_path / "clone"
    shutil.copytree(current_dir, temp_dir, dirs_exist_ok=True)
    os.chdir(temp_dir)
    yield temp_dir
    os.chdir(current_dir)


@pytest.fixture
def setup_argparse(monkeypatch, request, clone_current_directory_to_temp):
    config_path = clone_current_directory_to_temp / request.param.get("config_path", "configs/test_multi_config.json")
    args = ['program_name', '--use_config', str(config_path)]
    if "use_grayscale" in request.param:
        args.extend(['--use_grayscale', request.param.get("use_grayscale")])
    if "scaling_factor" in request.param:
        args.extend(['--scaling_factor', str(request.param.get("scaling_factor"))])

    monkeypatch.setattr('sys.argv', args)

@pytest.mark.parametrize('setup_argparse', [
    {"use_grayscale": "true"},
    {"use_grayscale": "false"},
    {"scaling_factor": "0.25"},
    {"scaling_factor": "0.5"},
    {"scaling_factor": "1"}
], indirect=["setup_argparse"])
def test_main_with_config(setup_argparse):
    main()
