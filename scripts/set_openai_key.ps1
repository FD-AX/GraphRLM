param(
    [string]$EnvFile = ".env.local"
)

$secure = Read-Host "OPENAI_API_KEY" -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
try {
    $plain = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    if ([string]::IsNullOrWhiteSpace($plain)) {
        throw "OPENAI_API_KEY is empty"
    }
    $content = "OPENAI_API_KEY=$plain"
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText((Resolve-Path -LiteralPath ".").Path + [System.IO.Path]::DirectorySeparatorChar + $EnvFile, $content, $utf8NoBom)
    Write-Host "Wrote $EnvFile. This file is ignored by git."
}
finally {
    if ($bstr -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    }
}
