param(
    [string]$MarkdownPath = "docs/project2_long_term_roadmap.md",
    [string]$HtmlPath = "docs/project2_long_term_roadmap.html",
    [string]$PdfPath = "docs/project2_long_term_roadmap.pdf"
)

$ErrorActionPreference = "Stop"

function Convert-InlineMarkdown {
    param([string]$Text)

    $encoded = [System.Net.WebUtility]::HtmlEncode($Text)
    $encoded = [regex]::Replace($encoded, '`([^`]+)`', '<code>$1</code>')
    $encoded = [regex]::Replace($encoded, '\*\*([^*]+)\*\*', '<strong>$1</strong>')
    $encoded = [regex]::Replace(
        $encoded,
        '\[([^\]]+)\]\(([^)]+)\)',
        '<a href="$2">$1</a>'
    )
    return $encoded
}

function Convert-TableCells {
    param(
        [string]$Line,
        [string]$Tag
    )

    $cells = $Line.Trim().Trim('|').Split('|')
    $rendered = foreach ($cell in $cells) {
        "<$Tag>$(Convert-InlineMarkdown $cell.Trim())</$Tag>"
    }
    return "<tr>$($rendered -join '')</tr>"
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$markdownFullPath = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $MarkdownPath))
$htmlFullPath = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $HtmlPath))
$pdfFullPath = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $PdfPath))

$lines = Get-Content -LiteralPath $markdownFullPath -Encoding UTF8
$body = [System.Collections.Generic.List[string]]::new()
$inCode = $false
$inUnorderedList = $false
$inOrderedList = $false
$inTable = $false
$tableHeaderPending = $false
$firstHeading = $true

foreach ($line in $lines) {
    if ($line -match '^```') {
        if ($inCode) {
            $body.Add('</code></pre>')
            $inCode = $false
        }
        else {
            $body.Add('<pre><code>')
            $inCode = $true
        }
        continue
    }

    if ($inCode) {
        $body.Add([System.Net.WebUtility]::HtmlEncode($line))
        continue
    }

    $isTableRow = $line.Trim().StartsWith('|') -and $line.Trim().EndsWith('|')
    $isSeparator = $line -match '^\|[\s\-:|]+\|$'

    if ($isTableRow) {
        if (-not $inTable) {
            if ($inUnorderedList) { $body.Add('</ul>'); $inUnorderedList = $false }
            if ($inOrderedList) { $body.Add('</ol>'); $inOrderedList = $false }
            $body.Add('<table>')
            $body.Add('<thead>')
            $body.Add((Convert-TableCells -Line $line -Tag 'th'))
            $tableHeaderPending = $true
            $inTable = $true
            continue
        }

        if ($isSeparator -and $tableHeaderPending) {
            $body.Add('</thead><tbody>')
            $tableHeaderPending = $false
            continue
        }

        if ($tableHeaderPending) {
            $body.Add('</thead><tbody>')
            $tableHeaderPending = $false
        }
        $body.Add((Convert-TableCells -Line $line -Tag 'td'))
        continue
    }

    if ($inTable) {
        if ($tableHeaderPending) { $body.Add('</thead><tbody>') }
        $body.Add('</tbody></table>')
        $inTable = $false
        $tableHeaderPending = $false
    }

    if ([string]::IsNullOrWhiteSpace($line)) {
        if ($inUnorderedList) { $body.Add('</ul>'); $inUnorderedList = $false }
        if ($inOrderedList) { $body.Add('</ol>'); $inOrderedList = $false }
        continue
    }

    if ($line -match '^####\s+(.+)$') {
        $body.Add("<h4>$(Convert-InlineMarkdown $Matches[1])</h4>")
        continue
    }
    if ($line -match '^###\s+(.+)$') {
        $body.Add("<h3>$(Convert-InlineMarkdown $Matches[1])</h3>")
        continue
    }
    if ($line -match '^##\s+(.+)$') {
        $body.Add("<h2>$(Convert-InlineMarkdown $Matches[1])</h2>")
        continue
    }
    if ($line -match '^#\s+(.+)$') {
        $className = if ($firstHeading) { ' class="document-title"' } else { '' }
        $body.Add("<h1$className>$(Convert-InlineMarkdown $Matches[1])</h1>")
        $firstHeading = $false
        continue
    }
    if ($line -match '^>\s?(.+)$') {
        $body.Add("<blockquote>$(Convert-InlineMarkdown $Matches[1])</blockquote>")
        continue
    }
    if ($line -match '^---+$') {
        $body.Add('<hr>')
        continue
    }
    if ($line -match '^-\s+(.+)$') {
        if ($inOrderedList) { $body.Add('</ol>'); $inOrderedList = $false }
        if (-not $inUnorderedList) { $body.Add('<ul>'); $inUnorderedList = $true }
        $body.Add("<li>$(Convert-InlineMarkdown $Matches[1])</li>")
        continue
    }
    if ($line -match '^\d+\.\s+(.+)$') {
        if ($inUnorderedList) { $body.Add('</ul>'); $inUnorderedList = $false }
        if (-not $inOrderedList) { $body.Add('<ol>'); $inOrderedList = $true }
        $body.Add("<li>$(Convert-InlineMarkdown $Matches[1])</li>")
        continue
    }

    if ($inUnorderedList) { $body.Add('</ul>'); $inUnorderedList = $false }
    if ($inOrderedList) { $body.Add('</ol>'); $inOrderedList = $false }
    $body.Add("<p>$(Convert-InlineMarkdown $line)</p>")
}

if ($inCode) { $body.Add('</code></pre>') }
if ($inUnorderedList) { $body.Add('</ul>') }
if ($inOrderedList) { $body.Add('</ol>') }
if ($inTable) {
    if ($tableHeaderPending) { $body.Add('</thead><tbody>') }
    $body.Add('</tbody></table>')
}

$html = @"
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Project 2 长期完善路线</title>
<style>
@page { size: A4; margin: 16mm 15mm 18mm; }
* { box-sizing: border-box; }
html { color: #18202a; background: #fff; }
body {
  max-width: 180mm;
  margin: 0 auto;
  font-family: "Microsoft YaHei", "SimSun", sans-serif;
  font-size: 10.5pt;
  line-height: 1.68;
}
.document-title {
  margin: 46mm 0 12mm;
  padding: 14mm 10mm;
  border-top: 4px solid #2457a7;
  border-bottom: 1px solid #b9c8de;
  color: #17365d;
  font-size: 27pt;
  text-align: center;
  letter-spacing: 1px;
}
h1, h2, h3, h4 { color: #17365d; line-height: 1.32; page-break-after: avoid; }
h2 {
  margin-top: 10mm;
  padding-bottom: 2.5mm;
  border-bottom: 1px solid #9fb6d5;
  font-size: 18pt;
  break-before: page;
}
h2:first-of-type { break-before: auto; }
h3 { margin: 7mm 0 3mm; font-size: 14pt; }
h4 { margin: 5mm 0 2mm; font-size: 11.5pt; }
p { margin: 2.2mm 0; }
blockquote {
  margin: 4mm 0;
  padding: 3mm 4mm;
  border-left: 4px solid #4c78b8;
  background: #f3f7fc;
  color: #33475b;
}
ul, ol { margin: 2mm 0 3mm 7mm; padding-left: 5mm; }
li { margin: 1.1mm 0; }
code {
  font-family: Consolas, "Microsoft YaHei", monospace;
  padding: .2mm 1mm;
  border-radius: 2px;
  background: #edf2f7;
  color: #7a2333;
}
pre {
  margin: 4mm 0;
  padding: 4mm;
  border: 1px solid #c8d4e3;
  border-radius: 4px;
  background: #f6f8fa;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  page-break-inside: avoid;
}
pre code { padding: 0; background: transparent; color: #1e2936; }
table {
  width: 100%;
  margin: 4mm 0 6mm;
  border-collapse: collapse;
  table-layout: fixed;
  font-size: 9.2pt;
}
thead { display: table-header-group; }
tr { page-break-inside: avoid; }
th, td {
  padding: 2.2mm 2.5mm;
  border: 1px solid #bdc9d8;
  vertical-align: top;
  overflow-wrap: anywhere;
}
th { color: #17365d; background: #e8f0fa; text-align: left; }
tr:nth-child(even) td { background: #f8fafc; }
a { color: #2457a7; text-decoration: none; overflow-wrap: anywhere; }
hr { margin: 7mm 0; border: 0; border-top: 1px solid #c8d4e3; }
strong { color: #142f50; }
.footer-note {
  margin-top: 12mm;
  padding-top: 3mm;
  border-top: 1px solid #c8d4e3;
  color: #64748b;
  font-size: 8.5pt;
  text-align: center;
}
</style>
</head>
<body>
$($body -join "`n")
<div class="footer-note">Project 2 长期完善路线 · v1.1 · 2026-10-01</div>
</body>
</html>
"@

[System.IO.File]::WriteAllText($htmlFullPath, $html, [System.Text.UTF8Encoding]::new($false))

$browserCandidates = @(
    'C:\Program Files\Google\Chrome\Application\chrome.exe',
    'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
    'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
    'C:\Program Files\Microsoft\Edge\Application\msedge.exe'
)
$browserPath = $browserCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $browserPath) {
    throw '未找到可用于打印 PDF 的 Edge 或 Chrome。'
}

$fileUri = [System.Uri]::new($htmlFullPath).AbsoluteUri
$profilePath = Join-Path $env:TEMP ("project2-roadmap-pdf-" + [guid]::NewGuid().ToString('N'))
$arguments = @(
    '--headless=new',
    '--no-sandbox',
    '--disable-gpu',
    '--disable-software-rasterizer',
    '--disable-gpu-compositing',
    '--disable-accelerated-2d-canvas',
    '--no-pdf-header-footer',
    "--user-data-dir=$profilePath",
    "--print-to-pdf=$pdfFullPath",
    $fileUri
)

if (Test-Path -LiteralPath $pdfFullPath) {
    Remove-Item -LiteralPath $pdfFullPath -Force
}

& $browserPath $arguments
$deadline = (Get-Date).AddSeconds(30)
while (-not (Test-Path -LiteralPath $pdfFullPath) -and (Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 250
}

if (-not (Test-Path -LiteralPath $pdfFullPath)) {
    throw '浏览器执行完成，但没有生成 PDF。'
}

$pdfInfo = Get-Item -LiteralPath $pdfFullPath
Write-Output "HTML: $htmlFullPath"
Write-Output "PDF:  $pdfFullPath"
Write-Output "SIZE: $($pdfInfo.Length) bytes"
