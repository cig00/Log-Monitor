const folderInput = document.querySelector("#folderInput");
const messageInput = document.querySelector("#messageInput");
const exactInput = document.querySelector("#exactInput");
const caseInput = document.querySelector("#caseInput");
const searchButton = document.querySelector("#searchButton");
const commitsButton = document.querySelector("#commitsButton");
const reportButton = document.querySelector("#reportButton");
const publishButton = document.querySelector("#publishButton");
const copilotButton = document.querySelector("#copilotButton");
const jiraKeyInput = document.querySelector("#jiraKeyInput");
const notesInput = document.querySelector("#notesInput");
const fileCount = document.querySelector("#fileCount");
const matchCount = document.querySelector("#matchCount");
const firstDate = document.querySelector("#firstDate");
const commitCount = document.querySelector("#commitCount");
const scanStatus = document.querySelector("#scanStatus");
const resultPanel = document.querySelector("#resultPanel");
const commitStatus = document.querySelector("#commitStatus");
const commitPanel = document.querySelector("#commitPanel");
const reportStatus = document.querySelector("#reportStatus");
const jiraStatus = document.querySelector("#jiraStatus");
const reportPanel = document.querySelector("#reportPanel");

let selectedFiles = [];
let firstAppearanceDate = null;
let firstAppearanceMatch = null;
let task2Commits = [];
let latestReport = "";

const datePatterns = [
  {
    regex:
      /\b(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{2}):(\d{2})(?::(\d{2})(?:[.,](\d{1,6}))?)?(?:\s*(Z|[+-]\d{2}:?\d{2}))?)?\b/,
    build: (match) => buildDate(match, 1, 2, 3, 4, 5, 6, 7, 8),
  },
  {
    regex:
      /\b(\d{4})\/(\d{2})\/(\d{2})(?:[ T](\d{2}):(\d{2})(?::(\d{2})(?:[.,](\d{1,6}))?)?)?\b/,
    build: (match) => buildDate(match, 1, 2, 3, 4, 5, 6, 7),
  },
  {
    regex:
      /\b(\d{2})\/(\d{2})\/(\d{4})(?:[ T](\d{2}):(\d{2})(?::(\d{2})(?:[.,](\d{1,6}))?)?)?\b/,
    build: (match) => buildDate(match, 3, 1, 2, 4, 5, 6, 7),
  },
  {
    regex: /\b(\d{4})(\d{2})(\d{2})(?:[_-]?(\d{2})(\d{2})(\d{2}))?\b/,
    build: (match) => buildDate(match, 1, 2, 3, 4, 5, 6),
  },
];

folderInput.addEventListener("change", () => {
  selectedFiles = Array.from(folderInput.files || []).filter((file) => !file.name.startsWith("."));
  firstAppearanceDate = null;
  firstAppearanceMatch = null;
  task2Commits = [];
  fileCount.textContent = selectedFiles.length.toLocaleString();
  matchCount.textContent = "0";
  commitCount.textContent = "0";
  firstDate.textContent = "-";
  commitsButton.disabled = true;
  reportButton.disabled = true;
  publishButton.disabled = true;
  copilotButton.disabled = true;
  scanStatus.textContent = selectedFiles.length
    ? `${selectedFiles.length.toLocaleString()} files ready`
    : "No folder selected";
  resultPanel.className = "result-panel empty";
  resultPanel.textContent = selectedFiles.length
    ? "Enter a log message and run the search."
    : "Choose a folder of logs, enter a message, then run the search.";
  commitStatus.textContent = "Run Task 1 first";
  commitPanel.className = "result-panel empty";
  commitPanel.textContent =
    "After Task 1 finds a date, run Task 2 to list commits from that date back seven days.";
  resetReport("Run Task 2 first");
});

searchButton.addEventListener("click", async () => {
  const message = messageInput.value.trim();

  if (!selectedFiles.length) {
    showEmpty("Choose a log folder first.");
    return;
  }

  if (!message) {
    showEmpty("Enter a log message to search for.");
    return;
  }

  searchButton.disabled = true;
  scanStatus.textContent = "Scanning...";
  resultPanel.className = "result-panel empty";
  resultPanel.textContent = "Reading log files.";

  try {
    const matches = await scanFiles(selectedFiles, message, {
      exact: exactInput.checked,
      caseSensitive: caseInput.checked,
    });

    matches.sort((a, b) => {
      if (a.timestamp && b.timestamp) return a.timestamp - b.timestamp;
      if (a.timestamp) return -1;
      if (b.timestamp) return 1;
      return a.file.localeCompare(b.file) || a.lineNumber - b.lineNumber;
    });

    renderResults(matches);
  } catch (error) {
    showEmpty(`Search failed: ${error.message}`);
    scanStatus.textContent = "Search failed";
  } finally {
    searchButton.disabled = false;
  }
});

reportButton.addEventListener("click", () => {
  renderJiraReport();
});

publishButton.addEventListener("click", async () => {
  await publishJiraComment();
});

copilotButton.addEventListener("click", async () => {
  const prompt = buildCopilotPrompt();

  try {
    await navigator.clipboard.writeText(prompt);
    jiraStatus.textContent = "Copilot prompt copied";
  } catch {
    reportPanel.className = "result-panel";
    reportPanel.textContent = prompt;
    jiraStatus.textContent = "Copy failed; prompt shown";
  }
});

commitsButton.addEventListener("click", async () => {
  if (!firstAppearanceDate) {
    showCommitEmpty("Run Task 1 first so there is a date to search from.");
    return;
  }

  commitsButton.disabled = true;
  commitStatus.textContent = "Scanning commits...";
  commitPanel.className = "result-panel empty";
  commitPanel.textContent = "Reading log files.";

  try {
    const commits = await scanCommits(selectedFiles, firstAppearanceDate);
    renderCommits(commits, firstAppearanceDate);
  } catch (error) {
    showCommitEmpty(`Commit search failed: ${error.message}`);
    commitStatus.textContent = "Search failed";
  } finally {
    commitsButton.disabled = false;
  }
});

async function scanFiles(files, message, options) {
  const normalizedMessage = options.caseSensitive ? message : message.toLowerCase();
  const matches = [];

  for (const file of files) {
    if (!isProbablyText(file)) continue;

    const text = await file.text();
    const lines = text.split(/\r?\n/);
    const fallbackDate = extractDate(file.webkitRelativePath || file.name);

    lines.forEach((line, index) => {
      const comparableLine = options.caseSensitive ? line : line.toLowerCase();
      const found = options.exact
        ? comparableLine.trim() === normalizedMessage
        : comparableLine.includes(normalizedMessage);

      if (!found) return;

      const lineDate = extractDate(line);
      matches.push({
        file: file.webkitRelativePath || file.name,
        line,
        lineNumber: index + 1,
        timestamp: lineDate?.date || fallbackDate?.date || null,
        dateSource: lineDate ? "line" : fallbackDate ? "filename" : "unknown",
      });
    });
  }

  return matches;
}

function renderResults(matches) {
  matchCount.textContent = matches.length.toLocaleString();
  commitCount.textContent = "0";
  task2Commits = [];
  commitStatus.textContent = "Run Task 1 first";
  commitPanel.className = "result-panel empty";
  commitPanel.textContent =
    "After Task 1 finds a date, run Task 2 to list commits from that date back seven days.";
  resetReport("Run Task 2 first");

  if (!matches.length) {
    firstAppearanceDate = null;
    firstAppearanceMatch = null;
    commitsButton.disabled = true;
    firstDate.textContent = "-";
    scanStatus.textContent = "No matches";
    showEmpty("No matching log message was found in the selected folder.");
    return;
  }

  const earliest = matches.find((match) => match.timestamp);
  firstAppearanceDate = earliest?.timestamp || null;
  firstAppearanceMatch = earliest || null;
  firstDate.textContent = earliest ? formatDate(earliest.timestamp) : "Unknown";
  commitsButton.disabled = !earliest;
  commitStatus.textContent = earliest ? "Ready" : "No searchable date";
  scanStatus.textContent = `${matches.length.toLocaleString()} matches found`;

  resultPanel.className = "result-panel";
  resultPanel.replaceChildren();

  if (!earliest) {
    const notice = document.createElement("div");
    notice.className = "notice";
    notice.textContent =
      "Matches were found, but no recognizable date was found on those lines or filenames.";
    resultPanel.append(notice);
  }

  const summary = document.createElement("div");
  summary.className = "match";
  summary.innerHTML = earliest
    ? `<strong>First appeared: ${escapeHtml(formatDate(earliest.timestamp))}</strong>
       <span class="meta">${escapeHtml(earliest.file)}:${earliest.lineNumber} - date from ${earliest.dateSource}</span>
       <code>${escapeHtml(earliest.line)}</code>`
    : `<strong>First matching line has no date</strong>
       <span class="meta">${escapeHtml(matches[0].file)}:${matches[0].lineNumber}</span>
       <code>${escapeHtml(matches[0].line)}</code>`;
  resultPanel.append(summary);

  const recentMatches = matches.slice(1, 8);
  recentMatches.forEach((match) => {
    const item = document.createElement("div");
    item.className = "match";
    item.innerHTML = `<span class="meta">${escapeHtml(match.file)}:${match.lineNumber}${
      match.timestamp ? ` - ${escapeHtml(formatDate(match.timestamp))}` : ""
    }</span><code>${escapeHtml(match.line)}</code>`;
    resultPanel.append(item);
  });
}

async function scanCommits(files, endDate) {
  const startDate = new Date(endDate);
  startDate.setDate(startDate.getDate() - 7);

  const commitsByHash = new Map();

  for (const file of files) {
    if (!isProbablyText(file)) continue;

    const text = await file.text();
    const lines = text.split(/\r?\n/);
    const fallbackDate = extractDate(file.webkitRelativePath || file.name);

    lines.forEach((line, index) => {
      const lineDate = extractDate(line);
      const timestamp = lineDate?.date || fallbackDate?.date || null;

      if (!timestamp || timestamp < startDate || timestamp > endDate) return;

      const hashes = extractCommitHashes(line);
      hashes.forEach((hash) => {
        const existing = commitsByHash.get(hash);
        const item = {
          hash,
          file: file.webkitRelativePath || file.name,
          line,
          lineNumber: index + 1,
          timestamp,
          dateSource: lineDate ? "line" : "filename",
        };

        if (!existing || item.timestamp < existing.timestamp) {
          commitsByHash.set(hash, item);
        }
      });
    });
  }

  return Array.from(commitsByHash.values()).sort((a, b) => {
    if (a.timestamp && b.timestamp) return a.timestamp - b.timestamp;
    return a.hash.localeCompare(b.hash);
  });
}

function renderCommits(commits, endDate) {
  const startDate = new Date(endDate);
  startDate.setDate(startDate.getDate() - 7);

  commitCount.textContent = commits.length.toLocaleString();
  task2Commits = commits;
  commitStatus.textContent = `${formatShortDate(startDate)} to ${formatShortDate(endDate)}`;
  resetReport(commits.length ? "Select related commits" : "No commits found");

  if (!commits.length) {
    reportButton.disabled = true;
    copilotButton.disabled = true;
    showCommitEmpty("No commit hashes were found in the seven days before the Task 1 date.");
    return;
  }

  reportButton.disabled = false;
  copilotButton.disabled = false;

  commitPanel.className = "result-panel";
  commitPanel.replaceChildren();

  const summary = document.createElement("div");
  summary.className = "notice";
  summary.textContent = `Found ${commits.length.toLocaleString()} unique commits from ${formatDate(
    startDate,
  )} through ${formatDate(endDate)}.`;
  commitPanel.append(summary);

  commits.forEach((commit) => {
    const item = document.createElement("div");
    item.className = "match commit-choice";
    item.innerHTML = `<input type="checkbox" data-commit-hash="${escapeHtml(commit.hash)}" aria-label="Mark ${escapeHtml(
      commit.hash,
    )} as related" />
      <div>
        <strong>${escapeHtml(commit.hash)}</strong>
        <span class="meta">${escapeHtml(formatDate(commit.timestamp))} - ${escapeHtml(
          commit.file,
        )}:${commit.lineNumber} - date from ${commit.dateSource}</span>
        <code>${escapeHtml(commit.line)}</code>
      </div>`;
    commitPanel.append(item);
  });
}

function renderJiraReport() {
  if (!firstAppearanceMatch || !task2Commits.length) {
    resetReport("Run Task 2 first");
    return;
  }

  const relatedCommits = getSelectedRelatedCommits();
  const jiraKey = jiraKeyInput.value.trim();
  const report = buildJiraReport(jiraKey, relatedCommits);
  latestReport = report;

  jiraStatus.textContent = relatedCommits.length
    ? `${relatedCommits.length.toLocaleString()} related commit(s) referenced`
    : "No related commit referenced";
  reportStatus.textContent = "Report generated";
  publishButton.disabled = false;
  reportPanel.className = "result-panel";
  reportPanel.innerHTML = `<textarea class="report-output" readonly>${escapeHtml(report)}</textarea>`;
}

async function publishJiraComment() {
  const issueKey = jiraKeyInput.value.trim().toUpperCase();

  if (!/^[A-Z][A-Z0-9]+-\d+$/.test(issueKey)) {
    jiraStatus.textContent = "Use full issue key, for example KAN-123";
    return;
  }

  renderJiraReport();

  publishButton.disabled = true;
  jiraStatus.textContent = "Publishing to JIRA...";

  try {
    const response = await fetch("/api/jira/comment", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        issueKey,
        comment: latestReport,
      }),
    });
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.error || "JIRA request failed.");
    }

    jiraStatus.textContent = `JIRA comment added${payload.id ? `: ${payload.id}` : ""}`;
  } catch (error) {
    jiraStatus.textContent = error.message;
  } finally {
    publishButton.disabled = false;
  }
}

function buildJiraReport(jiraKey, relatedCommits) {
  const notes = notesInput.value.trim();
  const lines = [
    jiraKey ? `JIRA: ${jiraKey}` : "JIRA: Not provided",
    "",
    "Summary",
    `The searched log message first appeared on ${formatDate(firstAppearanceMatch.timestamp)}.`,
    "",
    "Evidence",
    `Log file: ${firstAppearanceMatch.file}:${firstAppearanceMatch.lineNumber}`,
    `Log line: ${firstAppearanceMatch.line}`,
    "",
    "Commit review",
  ];

  if (relatedCommits.length) {
    lines.push("Related commit(s) to reference in JIRA:");
    relatedCommits.forEach((commit) => {
      lines.push(
        `- ${commit.hash} (${formatDate(commit.timestamp)}, ${commit.file}:${commit.lineNumber})`,
      );
      lines.push(`  Evidence: ${commit.line}`);
    });
  } else {
    lines.push("No reviewed commit was confirmed as related, so no commit is referenced.");
  }

  lines.push("", "Investigation notes", notes || "No additional notes provided.");

  return lines.join("\n");
}

function buildCopilotPrompt() {
  const commits = task2Commits
    .map(
      (commit) =>
        `- ${commit.hash} | ${formatDate(commit.timestamp)} | ${commit.file}:${commit.lineNumber} | ${commit.line}`,
    )
    .join("\n");

  return [
    "Review these log findings and candidate commits.",
    "Decide which commit, if any, is related to the issue. Only reference a commit if the evidence supports it.",
    "",
    `Log message searched: ${messageInput.value.trim() || "Not provided"}`,
    `First appearance: ${
      firstAppearanceMatch
        ? `${formatDate(firstAppearanceMatch.timestamp)} at ${firstAppearanceMatch.file}:${firstAppearanceMatch.lineNumber}`
        : "Not found"
    }`,
    `First matching line: ${firstAppearanceMatch?.line || "Not found"}`,
    "",
    "Candidate commits from the prior week:",
    commits || "No commits found.",
    "",
    "Return:",
    "1. Related commit hash, if any.",
    "2. Why it is related.",
    "3. A concise JIRA report.",
  ].join("\n");
}

function getSelectedRelatedCommits() {
  const selectedHashes = Array.from(commitPanel.querySelectorAll("[data-commit-hash]:checked")).map(
    (input) => input.dataset.commitHash,
  );
  return task2Commits.filter((commit) => selectedHashes.includes(commit.hash));
}

function resetReport(status) {
  latestReport = "";
  reportStatus.textContent = status;
  jiraStatus.textContent = "No report generated";
  reportButton.disabled = true;
  publishButton.disabled = true;
  copilotButton.disabled = true;
  reportPanel.className = "result-panel empty";
  reportPanel.textContent = "Mark related commits from Task 2, then generate the JIRA-ready report.";
}

function extractCommitHashes(line) {
  const hashes = new Set();
  const regex = /\b[0-9a-f]{7,40}\b/gi;
  let match = regex.exec(line);

  while (match) {
    const hash = match[0].toLowerCase();
    if (/[a-f]/.test(hash)) hashes.add(hash);
    match = regex.exec(line);
  }

  return Array.from(hashes);
}

function extractDate(value) {
  for (const pattern of datePatterns) {
    const match = value.match(pattern.regex);
    if (!match) continue;

    const date = pattern.build(match);
    if (!Number.isNaN(date.getTime())) {
      return { date };
    }
  }

  return null;
}

function buildDate(match, yearIndex, monthIndex, dayIndex, hourIndex, minuteIndex, secondIndex, msIndex, tzIndex) {
  const year = Number(match[yearIndex]);
  const month = Number(match[monthIndex]);
  const day = Number(match[dayIndex]);
  const hour = Number(match[hourIndex] || 0);
  const minute = Number(match[minuteIndex] || 0);
  const second = Number(match[secondIndex] || 0);
  const millis = Number((match[msIndex] || "0").padEnd(3, "0").slice(0, 3));
  const tz = tzIndex ? match[tzIndex] : "";

  if (tz) {
    const offset = tz === "Z" ? "Z" : normalizeTimezone(tz);
    return new Date(
      `${year}-${pad(month)}-${pad(day)}T${pad(hour)}:${pad(minute)}:${pad(second)}.${String(
        millis,
      ).padStart(3, "0")}${offset}`,
    );
  }

  return new Date(year, month - 1, day, hour, minute, second, millis);
}

function normalizeTimezone(value) {
  return value.includes(":") ? value : `${value.slice(0, 3)}:${value.slice(3)}`;
}

function isProbablyText(file) {
  if (file.type.startsWith("text/")) return true;
  return /\.(log|txt|out|err|trace|csv|json|ndjson|xml|yaml|yml)$/i.test(file.name);
}

function showEmpty(message) {
  resultPanel.className = "result-panel empty";
  resultPanel.textContent = message;
}

function showCommitEmpty(message) {
  commitPanel.className = "result-panel empty";
  commitPanel.textContent = message;
}

function formatDate(date) {
  return new Intl.DateTimeFormat(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(date);
}

function formatShortDate(date) {
  return new Intl.DateTimeFormat(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
  }).format(date);
}

function pad(value) {
  return String(value).padStart(2, "0");
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
