Get-CimInstance Win32_Process -Filter "Name='powershell.exe' OR Name='pwsh.exe'" | ForEach-Object {
    [pscustomobject]@{
        Pid = $_.ProcessId
        Cmd = $_.CommandLine
    }
} | Format-Table -AutoSize -Wrap
