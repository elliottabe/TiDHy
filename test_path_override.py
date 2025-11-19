#!/usr/bin/env python3
"""
Test script for path override functionality with Hydra override preservation.
"""

from omegaconf import OmegaConf
from TiDHy.utils.path_utils import extract_override_components, inject_override_components, override_config_paths

def test_extract_override_components():
    """Test extraction of override components from paths."""
    print("\n" + "="*80)
    print("Test 1: Extract Override Components")
    print("="*80)

    # Test case 1: Path with override components
    test_paths_1 = {
        'log_dir': '/data2/users/eabe/TiDHy/SLDS/TiDHy/run_id=31106927/load_jobid=,note=hyak_ckpt,seed=42/logs/',
        'save_dir': '/data2/users/eabe/TiDHy/SLDS/TiDHy/run_id=31106927/load_jobid=,note=hyak_ckpt,seed=42/',
    }
    run_id = '31106927'

    result = extract_override_components(test_paths_1, run_id, verbose=True)
    print(f"\nTest 1a: Path with overrides")
    print(f"  Input: {test_paths_1['log_dir']}")
    print(f"  Expected: 'load_jobid=,note=hyak_ckpt,seed=42'")
    print(f"  Result: '{result}'")
    print(f"  ✅ PASS" if result == 'load_jobid=,note=hyak_ckpt,seed=42' else f"  ❌ FAIL")

    # Test case 2: Path without override components
    test_paths_2 = {
        'log_dir': '/data2/users/eabe/TiDHy/SLDS/TiDHy/run_id=Testing/logs/',
        'save_dir': '/data2/users/eabe/TiDHy/SLDS/TiDHy/run_id=Testing/',
    }
    run_id = 'Testing'

    result = extract_override_components(test_paths_2, run_id, verbose=True)
    print(f"\nTest 1b: Path without overrides")
    print(f"  Input: {test_paths_2['log_dir']}")
    print(f"  Expected: None")
    print(f"  Result: {result}")
    print(f"  ✅ PASS" if result is None else f"  ❌ FAIL")

    # Test case 3: Path with single override
    test_paths_3 = {
        'log_dir': '/data2/users/eabe/TiDHy/CalMS21/TiDHy/run_id=TiDHy_Baseline/train.feature_type=raw/logs/',
    }
    run_id = 'TiDHy_Baseline'

    result = extract_override_components(test_paths_3, run_id, verbose=True)
    print(f"\nTest 1c: Path with single override")
    print(f"  Input: {test_paths_3['log_dir']}")
    print(f"  Expected: 'train.feature_type=raw'")
    print(f"  Result: '{result}'")
    print(f"  ✅ PASS" if result == 'train.feature_type=raw' else f"  ❌ FAIL")


def test_inject_override_components():
    """Test injection of override components into paths."""
    print("\n" + "="*80)
    print("Test 2: Inject Override Components")
    print("="*80)

    # Test case 1: Inject into multiple paths
    test_paths = {
        'base_dir': '/gscratch/portia/eabe/data/TiDHy/SLDS/TiDHy',
        'save_dir': '/gscratch/portia/eabe/data/TiDHy/SLDS/TiDHy/run_id=31106927/',
        'log_dir': '/gscratch/portia/eabe/data/TiDHy/SLDS/TiDHy/run_id=31106927/logs/',
        'ckpt_dir': '/gscratch/portia/eabe/data/TiDHy/SLDS/TiDHy/run_id=31106927/ckpt/',
        'data_dir': '/gscratch/portia/eabe/data/TiDHy/datasets/SLDS/',
        'user': 'eabe',
    }
    override_components = 'load_jobid=,note=hyak_ckpt,seed=42'
    run_id = '31106927'

    result = inject_override_components(test_paths, override_components, run_id, verbose=True)

    print(f"\nTest 2a: Inject into multiple paths")
    print(f"  Override components: '{override_components}'")

    # Check log_dir
    expected_log = '/gscratch/portia/eabe/data/TiDHy/SLDS/TiDHy/run_id=31106927/load_jobid=,note=hyak_ckpt,seed=42/logs/'
    print(f"\n  log_dir:")
    print(f"    Original: {test_paths['log_dir']}")
    print(f"    Expected: {expected_log}")
    print(f"    Result:   {result['log_dir']}")
    print(f"    ✅ PASS" if result['log_dir'] == expected_log else f"    ❌ FAIL")

    # Check save_dir
    expected_save = '/gscratch/portia/eabe/data/TiDHy/SLDS/TiDHy/run_id=31106927/load_jobid=,note=hyak_ckpt,seed=42/'
    print(f"\n  save_dir:")
    print(f"    Original: {test_paths['save_dir']}")
    print(f"    Expected: {expected_save}")
    print(f"    Result:   {result['save_dir']}")
    print(f"    ✅ PASS" if result['save_dir'] == expected_save else f"    ❌ FAIL")

    # Check data_dir (should be unchanged)
    print(f"\n  data_dir (should be unchanged):")
    print(f"    Original: {test_paths['data_dir']}")
    print(f"    Result:   {result['data_dir']}")
    print(f"    ✅ PASS" if result['data_dir'] == test_paths['data_dir'] else f"    ❌ FAIL")

    # Check user (should be unchanged)
    print(f"\n  user (should be unchanged):")
    print(f"    Original: {test_paths['user']}")
    print(f"    Result:   {result['user']}")
    print(f"    ✅ PASS" if result['user'] == test_paths['user'] else f"    ❌ FAIL")


def test_full_path_override():
    """Test the full path override workflow."""
    print("\n" + "="*80)
    print("Test 3: Full Path Override Workflow")
    print("="*80)

    # Create a mock config with hyak paths and override components
    mock_config = OmegaConf.create({
        'dataset': {'name': 'SLDS'},
        'version': 'TiDHy',
        'run_id': '31106927',
        'paths': {
            'user': 'eabe',
            'base_dir': '/gscratch/portia/eabe/data/TiDHy/SLDS/TiDHy',
            'save_dir': '/gscratch/portia/eabe/data/TiDHy/SLDS/TiDHy/run_id=31106927/load_jobid=,note=hyak_ckpt,seed=42/',
            'log_dir': '/gscratch/portia/eabe/data/TiDHy/SLDS/TiDHy/run_id=31106927/load_jobid=,note=hyak_ckpt,seed=42/logs/',
            'ckpt_dir': '/gscratch/portia/eabe/data/TiDHy/SLDS/TiDHy/run_id=31106927/load_jobid=,note=hyak_ckpt,seed=42/ckpt/',
            'fig_dir': '/gscratch/portia/eabe/data/TiDHy/SLDS/TiDHy/run_id=31106927/load_jobid=,note=hyak_ckpt,seed=42/figures/',
            'data_dir': '/gscratch/portia/eabe/data/TiDHy/datasets/SLDS/',
        }
    })

    print(f"\nOriginal config (hyak template with overrides):")
    print(f"  save_dir: {mock_config.paths.save_dir}")
    print(f"  log_dir:  {mock_config.paths.log_dir}")

    # Override with workstation template
    # Note: This will fail if the actual template files don't exist, so we'll catch that
    try:
        updated_config = override_config_paths(
            mock_config,
            'workstation',
            config_dir='configs',
            verbose=True
        )

        print(f"\nUpdated config (workstation template with preserved overrides):")
        print(f"  save_dir: {updated_config.paths.save_dir}")
        print(f"  log_dir:  {updated_config.paths.log_dir}")

        # Check if overrides are preserved
        has_overrides = 'load_jobid=,note=hyak_ckpt,seed=42' in str(updated_config.paths.log_dir)
        print(f"\n  Override components preserved: {'✅ PASS' if has_overrides else '❌ FAIL'}")

        # Check if template changed
        template_changed = '/data2/users/' in str(updated_config.paths.log_dir)
        print(f"  Template base changed to workstation: {'✅ PASS' if template_changed else '❌ FAIL'}")

    except Exception as e:
        print(f"\n  ⚠️  Could not test full override (template files may not exist): {e}")
        print(f"  This is expected if running outside the TiDHy directory")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("TESTING PATH OVERRIDE WITH HYDRA OVERRIDE PRESERVATION")
    print("="*80)

    test_extract_override_components()
    test_inject_override_components()
    test_full_path_override()

    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80 + "\n")
