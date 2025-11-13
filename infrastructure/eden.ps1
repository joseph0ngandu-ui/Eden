function eden {
    <#
    .SYNOPSIS
    Eden Advanced Trading System CLI
    
    .DESCRIPTION
    PowerShell wrapper for the Eden Advanced Trading System
    
    .EXAMPLE
    eden --phase3 --ml-enabled --start-date 2025-10-07 --end-date 2025-10-14
    #>
    python "$PSScriptRoot\eden_runner.py" $args
}

# Export the function
Export-ModuleMember -Function eden