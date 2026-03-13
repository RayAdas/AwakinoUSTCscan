# PowerShell脚本 - 使用SCP复制文件到Windows Server

$remoteUser = "Administrator"
$remoteHost = "192.168.1.102"
$remoteBasePath = "/C:/Users/Administrator/Desktop/AwakinoUSTCscan"

$filesToCopy = @(
    "config.ini"
    "data/RebuildDatasets/real_912_20260312105301"
    "data/NpWaveData/20250716_151236"
    "models/unet3d_best_20260305103011.pth"
)

foreach ($item in $filesToCopy) {
    if (Test-Path $item) {
        Write-Host "复制: $item"
        
        # 对于目录，确保以斜杠结尾以便递归复制所有内容
        if (Test-Path $item -PathType Container) {
            # 复制目录及其所有内容
            scp -r "$item" "$remoteUser@$remoteHost`:$remoteBasePath/"
        } else {
            # 先确保目标目录存在
            $remoteDir = Split-Path $item -Parent
            if ($remoteDir) {
                ssh "$remoteUser@$remoteHost" "if not exist `"$remoteBasePath/$remoteDir`" mkdir `"$remoteBasePath/$remoteDir`""
            }
            # 复制文件
            scp "$item" "$remoteUser@$remoteHost`:$remoteBasePath/$item"
        }
    } else {
        Write-Host "警告: $item 不存在" -ForegroundColor Yellow
    }
}