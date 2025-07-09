"""
Calibration Preset Management System

This module handles saving and loading calibration presets for the surround view system.
Presets allow users to save optimal calibration parameters and reuse them across sessions.
"""

import json
import os
from typing import Dict, Optional, Tuple, Any
from pathlib import Path


class PresetManager:
    """
    Manages calibration presets for the surround view system.
    
    Features:
    - Save calibration parameters as presets
    - Load previously saved presets
    - Check for existing preset files
    - Automatic preset validation
    """
    
    def __init__(self, preset_dir: str = "presets"):
        """
        Initialize preset manager.
        
        Args:
            preset_dir: Directory to store preset files
        """
        self.preset_dir = Path(preset_dir)
        self.preset_dir.mkdir(exist_ok=True)
        
        # Default preset filename
        self.preset_file = self.preset_dir / "calibration_presets.json"
    
    def save_preset(self, camera_name: str, params: Dict[str, Any], apply_to_all: bool = False) -> bool:
        """
        Save calibration parameters as a preset.
        
        Args:
            camera_name: Name of the camera (front, back, left, right)
            params: Dictionary of calibration parameters
            apply_to_all: If True, save as universal preset for all cameras
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Load existing presets or create new structure
            presets = self._load_presets_file()
            
            preset_data = {
                'scale': params.get('scale', (0.6, 0.6)),
                'shift': params.get('shift', (0.0, 0.0)),
                'distortion': params.get('distortion', (-0.29, 0.11, -0.0003, 0.003)),
                'timestamp': self._get_timestamp()
            }
            
            if apply_to_all:
                # Save as universal preset AND copy to all camera types
                presets['universal'] = preset_data.copy()
                
                # Also save for each specific camera
                for cam in ['front', 'back', 'left', 'right']:
                    presets[cam] = preset_data.copy()
                
                # Write to file
                with open(self.preset_file, 'w') as f:
                    json.dump(presets, f, indent=2, ensure_ascii=False)
                
                print(f"✅ Universal preset saved for ALL cameras (based on {camera_name})")
                return True
            else:
                # Save only for specific camera
                presets[camera_name] = preset_data
                
                # Write to file
                with open(self.preset_file, 'w') as f:
                    json.dump(presets, f, indent=2, ensure_ascii=False)
                
                print(f"✅ Preset saved for {camera_name} camera only")
                return True
            
        except Exception as e:
            print(f"❌ Failed to save preset for {camera_name}: {e}")
            return False
    
    def load_preset(self, camera_name: str) -> Optional[Dict[str, Any]]:
        """
        Load calibration preset for a specific camera.
        Falls back to universal preset if camera-specific preset not found.
        
        Args:
            camera_name: Name of the camera (front, back, left, right)
            
        Returns:
            Dictionary of calibration parameters or None if not found
        """
        try:
            presets = self._load_presets_file()
            
            # First, try to load camera-specific preset
            if camera_name in presets:
                preset = presets[camera_name]
                print(f"✅ Loaded camera-specific preset for {camera_name} camera (saved: {preset.get('timestamp', 'unknown')})")
                return {
                    'scale': tuple(preset.get('scale', (0.6, 0.6))),
                    'shift': tuple(preset.get('shift', (0.0, 0.0))),
                    'distortion': tuple(preset.get('distortion', (-0.29, 0.11, -0.0003, 0.003)))
                }
            
            # If no camera-specific preset, try universal preset
            if 'universal' in presets:
                preset = presets['universal']
                print(f"✅ Loaded universal preset for {camera_name} camera (saved: {preset.get('timestamp', 'unknown')})")
                return {
                    'scale': tuple(preset.get('scale', (0.6, 0.6))),
                    'shift': tuple(preset.get('shift', (0.0, 0.0))),
                    'distortion': tuple(preset.get('distortion', (-0.29, 0.11, -0.0003, 0.003)))
                }
            
            # If no universal preset, try to use front camera preset for other cameras
            if camera_name != 'front' and 'front' in presets:
                preset = presets['front']
                print(f"✅ Loaded front camera preset for {camera_name} camera (saved: {preset.get('timestamp', 'unknown')})")
                return {
                    'scale': tuple(preset.get('scale', (0.6, 0.6))),
                    'shift': tuple(preset.get('shift', (0.0, 0.0))),
                    'distortion': tuple(preset.get('distortion', (-0.29, 0.11, -0.0003, 0.003)))
                }
            
            print(f"ℹ️  No preset found for {camera_name} camera")
            return None
                
        except Exception as e:
            print(f"❌ Failed to load preset for {camera_name}: {e}")
            return None
    
    def has_preset(self, camera_name: str) -> bool:
        """
        Check if a preset exists for the specified camera.
        
        Args:
            camera_name: Name of the camera
            
        Returns:
            True if preset exists, False otherwise
        """
        try:
            presets = self._load_presets_file()
            return camera_name in presets and len(presets[camera_name]) > 0
            
        except Exception:
            return False
    
    def has_any_presets(self) -> bool:
        """
        Check if any presets exist.
        
        Returns:
            True if any presets exist, False otherwise
        """
        try:
            return self.preset_file.exists() and self.preset_file.stat().st_size > 0
            
        except Exception:
            return False
    
    def list_available_presets(self) -> Dict[str, str]:
        """
        List all available presets with their timestamps.
        
        Returns:
            Dictionary mapping camera names to timestamps
        """
        try:
            presets = self._load_presets_file()
            result = {}
            
            for camera_name, data in presets.items():
                if isinstance(data, dict) and 'timestamp' in data:
                    result[camera_name] = data['timestamp']
            
            return result
            
        except Exception:
            return {}
    
    def delete_preset(self, camera_name: str) -> bool:
        """
        Delete a specific camera preset.
        
        Args:
            camera_name: Name of the camera
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            presets = self._load_presets_file()
            
            if camera_name in presets:
                del presets[camera_name]
                
                with open(self.preset_file, 'w') as f:
                    json.dump(presets, f, indent=2, ensure_ascii=False)
                
                print(f"✅ Deleted preset for {camera_name} camera")
                return True
            else:
                print(f"ℹ️  No preset found for {camera_name} camera")
                return False
                
        except Exception as e:
            print(f"❌ Failed to delete preset for {camera_name}: {e}")
            return False
    
    def clear_all_presets(self) -> bool:
        """
        Clear all presets.
        
        Returns:
            True if cleared successfully, False otherwise
        """
        try:
            if self.preset_file.exists():
                self.preset_file.unlink()
                print("✅ All presets cleared")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to clear presets: {e}")
            return False
    
    def _load_presets_file(self) -> Dict[str, Any]:
        """
        Load presets from file.
        
        Returns:
            Dictionary of presets or empty dict if file doesn't exist
        """
        if not self.preset_file.exists():
            return {}
        
        try:
            with open(self.preset_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        import datetime
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def get_preset_file_path(self) -> str:
        """Get the path to the preset file."""
        return str(self.preset_file.absolute())


# Global preset manager instance
_preset_manager = None


def get_preset_manager() -> PresetManager:
    """Get the global preset manager instance."""
    global _preset_manager
    if _preset_manager is None:
        _preset_manager = PresetManager()
    return _preset_manager


def save_calibration_preset(camera_name: str, params: Dict[str, Any], apply_to_all: bool = False) -> bool:
    """
    Convenience function to save calibration preset.
    
    Args:
        camera_name: Camera name
        params: Calibration parameters
        apply_to_all: If True, apply preset to all cameras
        
    Returns:
        True if saved successfully
    """
    return get_preset_manager().save_preset(camera_name, params, apply_to_all)


def load_calibration_preset(camera_name: str) -> Optional[Dict[str, Any]]:
    """
    Convenience function to load calibration preset.
    
    Args:
        camera_name: Camera name
        
    Returns:
        Calibration parameters or None
    """
    return get_preset_manager().load_preset(camera_name)


def has_calibration_preset(camera_name: str) -> bool:
    """
    Convenience function to check if preset exists.
    
    Args:
        camera_name: Camera name
        
    Returns:
        True if preset exists
    """
    return get_preset_manager().has_preset(camera_name) 