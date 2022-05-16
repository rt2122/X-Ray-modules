import paramiko
import getpass
import os
from scp import SCPClient
from typing import Tuple, Dict, List


def createSSHClient(server: str, user: str, password: str) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=server, username=user, password=password, look_for_keys=False,
                   allow_agent=False)
    return client


def load_tiles(tiles: Tuple[str], dirs_dict: Dict[str, List[str]], local_dir: str,
               host: str = '193.232.11.134', user: str = 'alisa') -> None:
    """
    dirs_dict = {"dir1" : ["file1.fits"], "dir2" : ["file2.fits.gz"]}
    For each tile create directory in localdir. From each dir from dirs_dict download
    selected files and place them in local tile directory.
    """
    password = getpass.getpass()
    ssh = createSSHClient(host, user, password)
    scp = SCPClient(ssh.get_transport())
    for tile in tiles:
        tile_dir = os.path.join(local_dir, tile)
        if not os.path.isdir(tile_dir):
            os.mkdir(tile_dir)
        for rem_dir, files in dirs_dict.items():
            if rem_dir.count('{') > 1:
                rem_dir = rem_dir.format(tile[:3], tile)
            else:
                rem_dir = rem_dir.format(tile)
            for file in files:
                remote_file = os.path.join(rem_dir, file.format(tile))
                scp.get(remote_file, local_path=tile_dir)
