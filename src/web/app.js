const MODES = {
  bullet: [
    { label: "1|0", minutes: 1, increment: 0 },
    { label: "2|1", minutes: 2, increment: 1 },
  ],
  blitz: [
    { label: "3|0", minutes: 3, increment: 0 },
    { label: "5|0", minutes: 5, increment: 0 },
    { label: "5|3", minutes: 5, increment: 3 },
  ],
  rapid: [
    { label: "10|0", minutes: 10, increment: 0 },
    { label: "15|10", minutes: 15, increment: 10 },
  ],
};

const PIECES = {
  white: { k: "♔", q: "♕", r: "♖", b: "♗", n: "♘", p: "♙" },
  black: { k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟" },
};
const FILES = "abcdefgh";
const RANKS = "87654321";
const $ = (selector) => document.querySelector(selector);

const ui = {
  mode: "blitz",
  control: MODES.blitz[1],
  playerColor: "white",
  state: null,
  model: { available: false, detail: "" },
  selected: null,
  flipped: false,
  busy: false,
  toastTimer: null,
};

function initialPreview() {
  const pieces = {};
  const backRank = ["r", "n", "b", "q", "k", "b", "n", "r"];
  for (let file = 0; file < 8; file += 1) {
    pieces[`${FILES[file]}8`] = { color: "black", type: backRank[file] };
    pieces[`${FILES[file]}7`] = { color: "black", type: "p" };
    pieces[`${FILES[file]}2`] = { color: "white", type: "p" };
    pieces[`${FILES[file]}1`] = { color: "white", type: backRank[file] };
  }
  return {
    started: false,
    pieces,
    legal_moves: [],
    history: [],
    player_color: ui.playerColor,
    to_move: "white",
    clocks_ms: { white: ui.control.minutes * 60_000, black: ui.control.minutes * 60_000 },
    last_move: null,
    in_check: null,
    result: null,
  };
}

function renderModeControls() {
  const locked = Boolean(ui.state?.started && !ui.state.result);
  document.querySelectorAll(".mode-tab").forEach((button) => {
    const selected = button.dataset.mode === ui.mode;
    button.classList.toggle("is-selected", selected);
    button.setAttribute("aria-selected", String(selected));
    button.disabled = locked;
  });
  const options = $("#time-options");
  options.innerHTML = MODES[ui.mode].map((control, index) => `
    <button class="time-option ${sameControl(control, ui.control) ? "is-selected" : ""}" type="button" data-option="${index}" aria-pressed="${sameControl(control, ui.control)}">${control.label}</button>
  `).join("");
  options.querySelectorAll(".time-option").forEach((button) => {
    button.disabled = locked;
    button.addEventListener("click", () => {
      ui.control = MODES[ui.mode][Number(button.dataset.option)];
      if (!ui.state?.started) {
        renderBoard();
        renderClocks();
      }
      renderModeControls();
    });
  });
}

function sameControl(a, b) {
  return a.minutes === b.minutes && a.increment === b.increment;
}

function renderBoard() {
  const state = ui.state || initialPreview();
  const board = $("#chessboard");
  const isBlackView = ui.flipped ? ui.playerColor !== "black" : ui.playerColor === "black";
  const rankOrder = isBlackView ? "12345678" : RANKS;
  const fileOrder = isBlackView ? "hgfedcba" : FILES;
  const files = [...fileOrder];
  const ranks = [...rankOrder];
  const legalFromSelected = (state.legal_moves || []).filter((move) => move.slice(0, 2) === ui.selected);
  const legalTargets = new Set(legalFromSelected.map((move) => move.slice(2, 4)));
  const lastMove = state.last_move || null;
  const currentColor = state.to_move || "white";
  const canMove = Boolean(state.started && !state.result && !ui.busy && currentColor === ui.playerColor && ui.model.available);
  const squares = [];

  ranks.forEach((rank) => {
    files.forEach((file) => {
      const square = `${file}${rank}`;
      const piece = state.pieces?.[square];
      const isLight = (FILES.indexOf(file) + Number(rank)) % 2 === 0;
      const isTarget = legalTargets.has(square);
      const classes = ["square", isLight ? "light" : "dark"];
      if (lastMove?.includes(square)) classes.push("last-move");
      if (ui.selected === square) classes.push("selected");
      if (isTarget) classes.push(piece ? "legal-capture" : "legal-empty");
      if (state.in_check === square) classes.push("in-check");
      const pieceGlyph = piece ? `<span class="piece piece-${piece.color}" aria-hidden="true">${PIECES[piece.color][piece.type]}</span>` : "";
      const description = piece ? `${piece.color} ${piece.type} on ${square}` : square;
      squares.push(`<button class="${classes.join(" ")}" type="button" role="gridcell" data-square="${square}" aria-label="${description}" aria-selected="${ui.selected === square}" ${canMove && piece?.color === ui.playerColor ? 'draggable="true"' : ""}>${pieceGlyph}</button>`);
    });
  });

  board.innerHTML = squares.join("");
  board.classList.toggle("is-disabled-game", !canMove);
  $("#rank-labels").innerHTML = ranks.map((rank) => `<span>${rank}</span>`).join("");
  $("#file-labels").innerHTML = files.map((file) => `<span>${file}</span>`).join("");
  board.querySelectorAll(".square").forEach((squareButton) => {
    squareButton.addEventListener("click", () => onSquareClick(squareButton.dataset.square));
    squareButton.addEventListener("dragstart", (event) => {
      const source = squareButton.dataset.square;
      const selectedPiece = state.pieces?.[source];
      if (!canMove || selectedPiece?.color !== ui.playerColor) {
        event.preventDefault();
        return;
      }
      event.dataTransfer.setData("text/plain", source);
      event.dataTransfer.effectAllowed = "move";
      ui.selected = source;
      squareButton.classList.add("selected");
    });
    squareButton.addEventListener("dragover", (event) => {
      if (canMove) event.preventDefault();
    });
    squareButton.addEventListener("drop", (event) => {
      event.preventDefault();
      const source = event.dataTransfer.getData("text/plain");
      if (source) submitMove(source, squareButton.dataset.square);
    });
  });
}

function onSquareClick(square) {
  if (ui.busy || !ui.model.available || !ui.state?.started || ui.state.result || ui.state.to_move !== ui.playerColor) return;
  const piece = ui.state.pieces?.[square];
  if (!ui.selected) {
    if (piece?.color === ui.playerColor) {
      ui.selected = square;
      renderBoard();
    }
    return;
  }
  if (ui.selected === square) {
    ui.selected = null;
    renderBoard();
    return;
  }
  if (piece?.color === ui.playerColor) {
    ui.selected = square;
    renderBoard();
    return;
  }
  const move = (ui.state.legal_moves || []).find((candidate) => candidate.slice(0, 2) === ui.selected && candidate.slice(2, 4) === square);
  if (move) submitMove(ui.selected, square, move[4] || "q");
  else {
    ui.selected = null;
    renderBoard();
  }
}

async function submitMove(from, to, promotion = "q") {
  if (ui.busy || !ui.state?.started) return;
  const candidate = ui.state.legal_moves?.find((move) => move.slice(0, 2) === from && move.slice(2, 4) === to);
  if (!candidate) {
    ui.selected = null;
    renderBoard();
    return;
  }
  ui.busy = true;
  ui.selected = null;
  updateGameControls();
  renderBoard();
  setTurnMessage("ChessNet is thinking…", "Your local model is choosing a move.", true);
  try {
    const response = await api("/api/move", { from, to, promotion: candidate.length === 5 ? promotion : null });
    acceptState(response.state);
  } catch (error) {
    showToast(error.message || "That move could not be made.");
    await refreshState();
  } finally {
    ui.busy = false;
    if (ui.state) acceptState(ui.state);
    updateGameControls();
    renderBoard();
  }
}

function renderMoveHistory(history = []) {
  const target = $("#move-list");
  $("#move-count").textContent = `${history.length} ${history.length === 1 ? "move" : "moves"}`;
  if (!history.length) {
    target.innerHTML = `<div class="empty-moves"><span>♙</span><p>Your game will appear here.<br /><small>Every great game starts somewhere.</small></p></div>`;
    return;
  }
  const byNumber = new Map();
  history.forEach((entry, index) => {
    if (!byNumber.has(entry.number)) byNumber.set(entry.number, { number: entry.number });
    byNumber.get(entry.number)[entry.color] = entry.san;
    byNumber.get(entry.number)[`${entry.color}Index`] = index;
  });
  const latestIndex = history.length - 1;
  target.innerHTML = [...byNumber.values()].map((row) => `
    <div class="move-row"><span class="move-number">${row.number}.</span><span class="move-san ${row.whiteIndex === latestIndex ? "latest" : ""}">${row.white || ""}</span><span class="move-san ${row.blackIndex === latestIndex ? "latest" : ""}">${row.black || ""}</span></div>
  `).join("");
  target.scrollTop = target.scrollHeight;
}

function formatClock(milliseconds) {
  const safeMs = Math.max(0, milliseconds || 0);
  const totalSeconds = Math.ceil(safeMs / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function renderClocks() {
  const state = ui.state || initialPreview();
  const previewClock = ui.control.minutes * 60_000;
  const whiteClock = state.started ? state.clocks_ms?.white ?? previewClock : previewClock;
  const blackClock = state.started ? state.clocks_ms?.black ?? previewClock : previewClock;
  const playerMs = ui.playerColor === "white" ? whiteClock : blackClock;
  const engineMs = ui.playerColor === "white" ? blackClock : whiteClock;
  const playerTurn = state.started && !state.result && state.to_move === ui.playerColor;
  const engineTurn = state.started && !state.result && state.to_move !== ui.playerColor;
  $("#player-clock").textContent = formatClock(playerMs);
  $("#engine-clock").textContent = formatClock(engineMs);
  $("#player-clock-box").classList.toggle("is-running", playerTurn);
  $("#engine-clock-box").classList.toggle("is-running", engineTurn || ui.busy);
  $("#player-clock-box").classList.toggle("is-low", playerMs <= 10_000 && state.started && !state.result);
  $("#engine-clock-box").classList.toggle("is-low", engineMs <= 10_000 && state.started && !state.result);
}

function setTurnMessage(title, subtitle, live = false, over = false) {
  $("#turn-title").textContent = title;
  $("#turn-subtitle").textContent = subtitle;
  $("#turn-card").classList.toggle("is-live", live);
  $("#turn-card").classList.toggle("is-over", over);
  $("#turn-icon").textContent = over ? "⚑" : live ? "◷" : "✳";
}

function acceptState(state) {
  if (!state) return;
  if (!state.pieces && !state.started) state = initialPreview();
  if (state.clocks_ms) {
    const running = state.started && !state.result ? state.to_move : null;
    state._baseClock = running ? state.clocks_ms[running] : 0;
    state._receivedAt = Date.now();
  }
  if (state.started && MODES[state.mode]) {
    ui.mode = state.mode;
    const [minutes, increment] = (state.time_control || "").split("|").map(Number);
    ui.control = MODES[state.mode].find((control) => control.minutes === minutes && control.increment === increment) || ui.control;
  }
  ui.state = state;
  ui.playerColor = state.player_color || ui.playerColor;
  if (!state.started) ui.flipped = false;
  $("#player-caption").textContent = `Playing as ${capitalize(ui.playerColor)}`;
  $("#game-title").textContent = state.mode ? `${capitalize(state.mode)} · ${state.time_control}` : "A friendly match";
  if (!state.started) {
    setTurnMessage(ui.model.available ? "Board is ready" : "Waiting for your model", ui.model.available ? "Choose a time and start a game." : "Connect a checkpoint to begin playing.");
  } else if (state.result) {
    const won = state.result.winner === ui.playerColor;
    const draw = !state.result.winner;
    setTurnMessage(draw ? "A well-played draw" : won ? "A lovely win" : "Game over", state.result.message || (draw ? "The game is a draw." : won ? "Nicely done." : "There is always another game."), false, true);
  } else if (ui.busy || state.to_move !== ui.playerColor) {
    setTurnMessage("ChessNet to move", "Your local model is considering the position.", true);
  } else {
    setTurnMessage("Your move", "Take your time. The clock is yours.", true);
  }
  const engineBadge = $("#engine-turn-badge");
  engineBadge.textContent = state.started
    ? state.result
      ? "FINISHED"
      : state.to_move === ui.playerColor
        ? "YOUR TURN"
        : "THINKING"
    : "READY";
  engineBadge.classList.toggle("is-live", state.started && !state.result);
  $("#resign-button").disabled = !state.started || Boolean(state.result) || ui.busy;
  document.querySelectorAll(".side-option").forEach((option) => {
    const selected = option.dataset.color === ui.playerColor;
    option.classList.toggle("is-selected", selected);
    option.setAttribute("aria-pressed", String(selected));
    option.disabled = Boolean(state.started && !state.result);
  });
  renderMoveHistory(state.history || []);
  renderModeControls();
  renderBoard();
  renderClocks();
  updateGameControls();
}

function updateGameControls() {
  const button = $("#new-game-button");
  button.disabled = !ui.model.available || ui.busy;
  button.classList.toggle("is-loading", ui.busy);
}

function setModelStatus(model) {
  ui.model = model || { available: false, detail: "Engine connection unavailable" };
  const online = Boolean(ui.model.available);
  $("#sidebar-engine-label").textContent = online ? "Engine ready" : "Model not loaded";
  $("#sidebar-engine-detail").textContent = online ? "Ready for a game whenever you are." : "No checkpoint connected yet.";
  $("#sidebar-engine-dot").classList.toggle("is-online", online);
  $("#sidebar-engine-dot").classList.toggle("is-offline", !online);
  $("#engine-status-dot").classList.toggle("is-online", online);
  $("#engine-status-dot").classList.toggle("is-offline", !online);
  $("#engine-caption").textContent = online ? "Your local policy model" : "Waiting for a model checkpoint";
  const badge = document.querySelector(".local-badge");
  badge?.classList.toggle("is-online", online);
  $("#dialog-model-detail").textContent = online
    ? `Connected checkpoint: ${ui.model.checkpoint || "ChessNet"} (${ui.model.device || "local device"}).`
    : ui.model.detail || "No checkpoint was detected in the usual checkpoints folder.";
  updateGameControls();
}

async function api(path, body) {
  const response = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  let payload;
  try { payload = await response.json(); } catch { payload = {}; }
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

async function refreshState() {
  try {
    const data = await api("/api/state");
    if (data.state) acceptState(data.state);
    else if (!ui.state) acceptState(initialPreview());
  } catch {
    if (!ui.state) acceptState(initialPreview());
  }
}

async function startGame() {
  if (!ui.model.available || ui.busy) return;
  ui.busy = true;
  updateGameControls();
  setTurnMessage("Setting the board…", "Getting a fresh game ready for you.", true);
  try {
    const data = await api("/api/new", {
      mode: ui.mode,
      minutes: ui.control.minutes,
      increment: ui.control.increment,
      player_color: ui.playerColor,
    });
    ui.flipped = false;
    acceptState(data.state);
    showToast(`${capitalize(ui.mode)} game started. Have a good one.`);
  } catch (error) {
    setTurnMessage("Could not start a game", error.message || "Check your engine setup.", false, true);
    showToast(error.message || "Could not start a game.");
  } finally {
    ui.busy = false;
    if (ui.state) acceptState(ui.state);
    updateGameControls();
    renderBoard();
  }
}

async function resignGame() {
  if (!ui.state?.started || ui.state.result || ui.busy) return;
  try {
    const data = await api("/api/resign", {});
    acceptState(data.state);
  } catch (error) {
    showToast(error.message || "Could not resign this game.");
  }
}

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("is-visible");
  clearTimeout(ui.toastTimer);
  ui.toastTimer = setTimeout(() => toast.classList.remove("is-visible"), 3400);
}

function capitalize(value = "") {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function bindControls() {
  document.querySelectorAll(".mode-tab").forEach((button) => {
    button.addEventListener("click", () => {
      ui.mode = button.dataset.mode;
      ui.control = MODES[ui.mode][Math.min(1, MODES[ui.mode].length - 1)];
      renderModeControls();
      if (!ui.state?.started) renderClocks();
    });
  });
  document.querySelectorAll(".side-option").forEach((button) => {
    button.addEventListener("click", () => {
      if (ui.state?.started && !ui.state.result) {
        showToast("Finish or start a new game to change sides.");
        return;
      }
      ui.playerColor = button.dataset.color;
      document.querySelectorAll(".side-option").forEach((option) => {
        const selected = option === button;
        option.classList.toggle("is-selected", selected);
        option.setAttribute("aria-pressed", String(selected));
      });
      if (ui.state && !ui.state.started) ui.state.player_color = ui.playerColor;
      $("#player-caption").textContent = `Playing as ${capitalize(ui.playerColor)}`;
      renderBoard();
    });
  });
  $("#new-game-button").addEventListener("click", startGame);
  $("#flip-board-button").addEventListener("click", () => {
    ui.flipped = !ui.flipped;
    renderBoard();
  });
  $("#resign-button").addEventListener("click", resignGame);
  $("#engine-help-button").addEventListener("click", () => $("#setup-dialog").showModal());
  $("#engine-help-top").addEventListener("click", () => $("#setup-dialog").showModal());
  $("#theme-button").addEventListener("click", () => {
    document.body.classList.toggle("soft-contrast");
    showToast(document.body.classList.contains("soft-contrast") ? "Soft contrast is on." : "Standard contrast is on.");
  });
}

async function initialize() {
  bindControls();
  renderModeControls();
  ui.state = initialPreview();
  renderBoard();
  renderClocks();
  try {
    const status = await api("/api/health");
    setModelStatus(status.model);
  } catch {
    setModelStatus({
      available: false,
      detail: "Open Hearth Chess with python -m whisker.serve to connect the local engine.",
    });
  }
  await refreshState();
  let syncingFlag = false;
  setInterval(() => {
    if (!ui.state?.started || ui.state.result || ui.busy) return;
    const elapsed = Date.now() - (ui.state._receivedAt || Date.now());
    const activeColor = ui.state.to_move;
    if (!activeColor) return;
    const clockKey = activeColor;
    ui.state.clocks_ms[clockKey] = Math.max(0, ui.state._baseClock - elapsed);
    renderClocks();
    if (ui.state.clocks_ms[clockKey] <= 0 && !syncingFlag) {
      syncingFlag = true;
      refreshState().finally(() => { syncingFlag = false; });
    }
  }, 200);
}

initialize();
