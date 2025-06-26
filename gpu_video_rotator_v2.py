# make sure cuda toolkit is installed
# install FFmpeg, e.g., https://www.gyan.dev/ffmpeg/builds/

import os
import subprocess
import sys
from pathlib import Path

# TODO: search "folder_path" and change the path of the videos

def detect_gpu_capabilities():
    """Detect available GPU hardware acceleration options."""
    gpu_options = {}
    
    # Test NVIDIA NVENC (requires NVIDIA GPU with NVENC support)
    # Install NVIDIA drivers and CUDA toolkit
    # FFmpeg should be compiled with --enable-nvenc
    try:
        result = subprocess.run(['ffmpeg', '-encoders'], 
                              capture_output=True, text=True, check=True)
        if 'h264_nvenc' in result.stdout:
            gpu_options['nvidia'] = {
                'encoder': 'h264_nvenc',
                'decoder': 'h264_cuvid',
                'hwaccel': 'cuda'
            }
    except:
        pass
    
    # Test Intel Quick Sync (requires Intel iGPU)
    # Install Intel Media SDK
    # Most pre-built FFmpeg includes Intel QSV support
    try:
        result = subprocess.run(['ffmpeg', '-encoders'], 
                              capture_output=True, text=True, check=True)
        if 'h264_qsv' in result.stdout:
            gpu_options['intel'] = {
                'encoder': 'h264_qsv',
                'decoder': 'h264_qsv',
                'hwaccel': 'qsv'
            }
    except:
        pass
    
    # Test AMD AMF (requires AMD GPU)
    # Install AMD drivers
    # FFmpeg should be compiled with --enable-amf
    try:
        result = subprocess.run(['ffmpeg', '-encoders'], 
                              capture_output=True, text=True, check=True)
        if 'h264_amf' in result.stdout:
            gpu_options['amd'] = {
                'encoder': 'h264_amf',
                'decoder': 'h264',
                'hwaccel': 'dxva2'  # or 'd3d11va' on newer systems
            }
    except:
        pass
    
    return gpu_options

def rotate_video_gpu(input_path, output_path, rotation, gpu_type=None):
    """
    Rotate video using GPU acceleration.
    
    Args:
        input_path (str): Input video path
        output_path (str): Output video path  
        rotation (str): '90cw', '90ccw', '180'
        gpu_type (str): 'nvidia', 'intel', 'amd', or None for auto-detect
    """
    gpu_options = detect_gpu_capabilities()
    
    if not gpu_options:
        print("No GPU acceleration detected, falling back to CPU")
        return rotate_video_cpu(input_path, output_path, rotation)
    
    # Select GPU type
    if gpu_type and gpu_type in gpu_options:
        selected_gpu = gpu_options[gpu_type]
    else:
        # Auto-select first available GPU
        selected_gpu = next(iter(gpu_options.values()))
        gpu_type = next(iter(gpu_options.keys()))
    
    print(f"Using {gpu_type.upper()} GPU acceleration")
    
    # Rotation filters
    rotation_filters = {
        '90cw': 'transpose=1',
        '90ccw': 'transpose=2', 
        '180': 'transpose=2,transpose=2'
    }
    
    if rotation not in rotation_filters:
        raise ValueError("Rotation must be '90cw', '90ccw', or '180'")
    
    # Build FFmpeg command with GPU acceleration
    if gpu_type == 'nvidia':
        cmd = [
            'ffmpeg',
            # '-hwaccel', 'cuda',
            # '-hwaccel_output_format', 'cuda', # can't a CPU-side filter (transpose) on a GPU surface
            '-i', input_path,
            '-vf', f'{rotation_filters[rotation]}',
            '-c:v', 'h264_nvenc',
            '-preset', 'fast',  # NVENC preset
            '-an',
            '-y', output_path
        ]
    elif gpu_type == 'intel':
        cmd = [
            'ffmpeg', 
            '-hwaccel', 'qsv',
            '-i', input_path,
            '-vf', f'{rotation_filters[rotation]}',
            '-c:v', 'h264_qsv',
            '-preset', 'fast',
            '-an',
            '-y', output_path
        ]
    elif gpu_type == 'amd':
        cmd = [
            'ffmpeg',
            '-hwaccel', 'dxva2',
            '-i', input_path, 
            '-vf', f'{rotation_filters[rotation]}',
            '-c:v', 'h264_amf',
            '-quality', 'speed',  # AMF quality preset
            '-an',
            '-y', output_path
        ]
    
    return subprocess.run(cmd, capture_output=True, text=True)

def rotate_video_cpu(input_path, output_path, rotation):
    """Fallback CPU rotation."""
    rotation_filters = {
        '90cw': 'transpose=1',
        '90ccw': 'transpose=2',
        '180': 'transpose=2,transpose=2'
    }
    
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-vf', rotation_filters[rotation],
        '-an',
        '-y', output_path
    ]
    
    return subprocess.run(cmd, capture_output=True, text=True)

def batch_rotate_videos_gpu(folder_path, rotation='90cw', gpu_type=None, output_suffix='_rotated'):
    """
    GPU-accelerated batch video rotation.
    
    Args:
        folder_path (str): Folder containing videos
        rotation (str): Rotation type
        gpu_type (str): 'nvidia', 'intel', 'amd', or None for auto
        output_suffix (str): Output filename suffix
    """
    # Check FFmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      capture_output=True, check=True)
    except:
        print("Error: FFmpeg not found")
        return
    
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist")
        return
    
    # Find video files (avoid duplicates on case-insensitive systems)
    video_files = []
    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in ['.mp4', '.mov']:
            video_files.append(file)
    
    if not video_files:
        print(f"No video files found in '{folder_path}'")
        return
    
    # Detect GPU capabilities
    gpu_options = detect_gpu_capabilities()
    if gpu_options:
        print("Available GPU acceleration:")
        for gpu_name in gpu_options:
            print(f"  - {gpu_name.upper()}")
        print()
    else:
        print("No GPU acceleration available, using CPU")
        print()
    
    print(f"Found {len(video_files)} video file(s)")
    print(f"Rotation: {rotation}")
    print("Starting batch rotation...\n")
    
    success_count = 0
    error_count = 0
    
    for video_file in video_files:
        # output_name = f"{video_file.stem}{output_suffix}{video_file.suffix}"
        output_name = f"{video_file.stem}{output_suffix}.mp4"
        output_path = folder / "flipped" /  output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing: {video_file.name}")
        
        try:
            result = rotate_video_gpu(str(video_file), str(output_path), 
                                    rotation, gpu_type)
            
            if result.returncode == 0:
                print(f"Success: {output_name}")
                success_count += 1
            else:
                print(f"Error processing {video_file.name}")
                if result.stderr:
                    print(f"  Error: {result.stderr[:200]}...")
                error_count += 1
                
        except Exception as e:
            print(f"Error processing {video_file.name}: {str(e)}")
            error_count += 1
        
        print()
    
    print(f"Batch rotation complete!")
    print(f"Successfully processed: {success_count}")
    print(f"Errors: {error_count}")

def main():
    """Main function with GPU options."""
    print("GPU-Accelerated Video Rotator")
    print("=" * 35)
    
    # Get folder path
    folder_path = 'F:/Test/CV/src/preprocess/raw'
    # folder_path = input("Enter folder path containing videos: ").strip()
    # if not folder_path:
        # folder_path = "."
    
    # Detect available GPUs
    gpu_options = detect_gpu_capabilities()
    
    # GPU selection
    gpu_type = None
    if gpu_options:
        print(f"\nAvailable GPU acceleration:")
        gpu_list = list(gpu_options.keys())
        for i, gpu_name in enumerate(gpu_list, 1):
            print(f"{i}. {gpu_name.upper()}")
        print(f"{len(gpu_list) + 1}. CPU only")
        
        while True:
            try:
                choice = int(input(f"\nSelect acceleration (1-{len(gpu_list) + 1}): "))
                if 1 <= choice <= len(gpu_list):
                    gpu_type = gpu_list[choice - 1]
                    break
                elif choice == len(gpu_list) + 1:
                    gpu_type = None
                    break
                else:
                    print("Invalid choice")
            except ValueError:
                print("Please enter a number")
    else:
        print("\nNo GPU acceleration available - using CPU")
    
    # Get rotation
    rotation = '180'
    # print("\nRotation options:")
    # print("1. 90° clockwise")
    # print("2. 90° counterclockwise")
    # print("3. 180°")
    
    # while True:
        # choice = input("\nSelect rotation (1-3): ").strip()
        # if choice == '1':
            # rotation = '90cw'
            # break
        # elif choice == '2':
            # rotation = '90ccw'
            # break
        # elif choice == '3':
            # rotation = '180'
            # break
        # else:
            # print("Invalid choice")
    
    # Get output suffix
    suffix = '_rotated'
    # suffix = input("\nOutput filename suffix (default: '_rotated'): ").strip()
    # if not suffix:
        # suffix = '_rotated'
    
    print()
    batch_rotate_videos_gpu(folder_path, rotation, gpu_type, suffix)

if __name__ == "__main__":
    main()
