Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ForEach-Object {
    [pscustomobject]@{
        Pid = $_.ProcessId
        Cmd = $_.CommandLine
    }
} | Format-Table -AutoSize -Wrap
