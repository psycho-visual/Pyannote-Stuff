import yt_dlp
import tempfile
import os



def download_audio_only(url, save_to_disk=False):
    if save_to_disk:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        output_dir = tempfile.mkdtemp()
    outtmpl_path = os.path.join(output_dir, 'audio.%(ext)s')
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': outtmpl_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }]
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    final_wav_path = os.path.join(output_dir, 'audio.wav')
    return output_dir, final_wav_path

if __name__ == '__main__':
    url = input('enter url: ')
    dir, wav_path = download_audio_only(url, save_to_disk=True)
    print(f'dir: {dir}')
    print(f'audio path: {wav_path}')