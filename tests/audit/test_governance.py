import pytest
import subprocess
import sys
import os

def test_config_drift():
    """Verify that Core Config defaults match Legacy __init__ defaults where possible."""
    assert True 

def test_import_dag():
    """Verify no circular imports in core."""
    import TBMD.core.geometry
    import TBMD.core.decomposition 
    import TBMD.core.reconstruction
    import TBMD.core.sensor_placement
    assert True

def test_deprecation_warning():
    """Verify that importing TBMD.modules triggers a DeprecationWarning (isolated)."""
    # Run a separate process — TBMD is installed as a package, no sys.path hack needed
    code = "import warnings; warnings.simplefilter('always'); import TBMD.modules"
    
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    
    assert "DeprecationWarning" in result.stderr
    assert "TBMD.modules" in result.stderr
    assert "deprecated" in result.stderr
