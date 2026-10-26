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

const PIECE_NAMES = {
  k: "king",
  q: "queen",
  r: "rook",
  b: "bishop",
  n: "knight",
  p: "pawn",
};
const SOUND_FILES = {
  move: "/assets/audio/move.mp3",
  capture: "/assets/audio/capture.mp3",
  castle: "/assets/audio/castle.mp3",
  check: "/assets/audio/check.mp3",
  promote: "/assets/audio/promote.mp3",
  gameend: "/assets/audio/gameend.mp3",
};
const FILES = "abcdefgh";
const RANKS = "87654321";
const PREMOVE_WINDOW_MS = 500;
const $ = (selector) => document.querySelector(selector);

const ui = {
  mode: "blitz",
  control: MODES.blitz[1],
  playerColor: "white",
  colorChoice: "white",
  state: null,
  model: { available: false, detail: "" },
  selected: null,
  flipped: false,
  busy: false,
  engineThinking: false,
  animating: false,
  animationTimer: null,
  moveTimeouts: [],
  pieceMoveGhosts: [],
  soundEnabled: true,
  audioPlayers: null,
  pendingMove: null,
  pendingDragGhost: null,
  premove: null,
  suppressNextMoveSound: false,
  drag: null,
  suppressBoardClick: false,
  startingGame: false,
  studyGame: null,
  studyTimer: null,
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
    button.setAttribute("aria-pressed", String(selected));
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

function parseUciMove(uci) {
  if (typeof uci !== "string" || !/^[a-h][1-8][a-h][1-8][qrbn]?$/.test(uci)) return null;
  return { from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] || null };
}

function premoveTargets(from, state = ui.state) {
  const piece = state?.pieces?.[from];
  if (!piece || piece.color !== ui.playerColor) return [];
  const file = FILES.indexOf(from[0]);
  const rank = Number(from[1]);
  const targets = [];
  const add = (nextFile, nextRank, sliding = false) => {
    while (nextFile >= 0 && nextFile < 8 && nextRank >= 1 && nextRank <= 8) {
      const square = `${FILES[nextFile]}${nextRank}`;
      const occupant = state.pieces?.[square];
      if (!occupant) targets.push(square);
      else {
        if (occupant.color !== piece.color) targets.push(square);
        break;
      }
      if (!sliding) break;
      nextFile += Math.sign(nextFile - file);
      nextRank += Math.sign(nextRank - rank);
    }
  };
  if (piece.type === "p") {
    const direction = piece.color === "white" ? 1 : -1;
    const nextRank = rank + direction;
    if (nextRank >= 1 && nextRank <= 8 && !state.pieces?.[`${FILES[file]}${nextRank}`]) {
      targets.push(`${FILES[file]}${nextRank}`);
      const startRank = piece.color === "white" ? 2 : 7;
      const doubleRank = rank + 2 * direction;
      if (rank === startRank && !state.pieces?.[`${FILES[file]}${doubleRank}`]) targets.push(`${FILES[file]}${doubleRank}`);
    }
    for (const offset of [-1, 1]) {
      const targetFile = file + offset;
      const target = `${FILES[targetFile]}${nextRank}`;
      if (targetFile >= 0 && targetFile < 8 && state.pieces?.[target]?.color !== piece.color && state.pieces?.[target]) targets.push(target);
    }
    return targets;
  }
  const knightOffsets = [[1, 2], [2, 1], [2, -1], [1, -2], [-1, -2], [-2, -1], [-2, 1], [-1, 2]];
  if (piece.type === "n") knightOffsets.forEach(([df, dr]) => add(file + df, rank + dr));
  if (piece.type === "k") {
    for (let df = -1; df <= 1; df += 1) {
      for (let dr = -1; dr <= 1; dr += 1) if (df || dr) add(file + df, rank + dr);
    }
  }
  const directions = [];
  if (["b", "q"].includes(piece.type)) directions.push([1, 1], [1, -1], [-1, 1], [-1, -1]);
  if (["r", "q"].includes(piece.type)) directions.push([1, 0], [-1, 0], [0, 1], [0, -1]);
  directions.forEach(([df, dr]) => add(file + df, rank + dr, true));
  return [...new Set(targets)];
}

function setAvatarPieces(topColor, bottomColor) {
  const topPiece = $(".opponent-avatar img");
  const bottomPiece = $(".self-avatar img");
  if (topPiece) topPiece.src = `/assets/pieces/wintrchess/standard/${topColor}_king.svg`;
  if (bottomPiece) bottomPiece.src = `/assets/pieces/wintrchess/standard/${bottomColor}_king.svg`;
}

function prepareAudioPlayers() {
  if (ui.audioPlayers || typeof Audio === "undefined") return ui.audioPlayers;
  ui.audioPlayers = Object.fromEntries(Object.entries(SOUND_FILES).map(([name, url]) => {
    const player = new Audio(url);
    player.preload = "auto";
    player.load();
    return [name, player];
  }));
  return ui.audioPlayers;
}

function unlockAudio() {
  if (!ui.soundEnabled) return Promise.resolve(false);
  try {
    prepareAudioPlayers();
  } catch (error) {
    console.warn("Whisker could not preload move sounds.", error);
  }
  return Promise.resolve(Boolean(ui.audioPlayers));
}

function playMoveSound(sound = "move") {
  if (!ui.soundEnabled) return;
  try {
    const players = prepareAudioPlayers();
    const original = players?.[sound] || players?.move;
    if (!original) return;
    const player = original.cloneNode(true);
    player.volume = 0.8;
    const playback = player.play();
    playback?.catch((error) => {
      if (error.name !== "NotAllowedError") console.warn(`Whisker could not play ${sound} sound.`, error);
    });
  } catch (error) {
    console.warn(`Whisker could not play ${sound} sound.`, error);
  }
}

function animateBoardMoves(moves, previousState, currentState) {
  if (!moves.length) return;
  const board = $("#chessboard");
  const pieces = { ...(previousState?.pieces || {}) };
  const reducedMotion = Boolean(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);

  moves.forEach((move, index) => {
    const movingPiece = pieces[move.from];
    const targetPiece = pieces[move.to];
    const isEnPassant = movingPiece?.type === "p" && move.from[0] !== move.to[0] && !targetPiece;
    const captured = Boolean(targetPiece) || isEnPassant;
    const isCastle = movingPiece?.type === "k" && Math.abs(FILES.indexOf(move.from[0]) - FILES.indexOf(move.to[0])) === 2;
    const isPromotion = movingPiece?.type === "p" && move.promotion && ["1", "8"].includes(move.to[1]);
    const destination = board.querySelector(`[data-square="${move.to}"] .piece-svg`);
    const sourceSquare = board.querySelector(`[data-square="${move.from}"]`);
    const targetSquare = board.querySelector(`[data-square="${move.to}"]`);

    if (!reducedMotion && movingPiece && sourceSquare && targetSquare && typeof Element.prototype.animate === "function") {
      const sourceRect = sourceSquare.getBoundingClientRect();
      const targetRect = targetSquare.getBoundingClientRect();
      const piece = isPromotion ? { ...movingPiece, type: move.promotion } : movingPiece;
      const ghost = document.createElement("img");
      ghost.src = `/assets/pieces/cburnett/${piece.color}_${PIECE_NAMES[piece.type]}.svg`;
      ghost.alt = "";
      ghost.draggable = false;
      ghost.className = "piece-svg piece-move-ghost";
      const pieceSize = Math.min(sourceRect.width, sourceRect.height) * .88;
      const startX = sourceRect.left + (sourceRect.width - pieceSize) / 2;
      const startY = sourceRect.top + (sourceRect.height - pieceSize) / 2;
      const endX = targetRect.left + (targetRect.width - pieceSize) / 2;
      const endY = targetRect.top + (targetRect.height - pieceSize) / 2;
      ghost.style.width = `${pieceSize}px`;
      ghost.style.height = `${pieceSize}px`;
      ghost.style.transform = `translate3d(${startX}px, ${startY}px, 0)`;
      if (destination) destination.style.visibility = "hidden";
      document.body.append(ghost);
      ui.pieceMoveGhosts.push(ghost);
      const animation = ghost.animate([
        { transform: `translate3d(${startX}px, ${startY}px, 0) scale(.97)` },
        { transform: `translate3d(${endX}px, ${endY}px, 0) scale(1)` },
      ], {
        duration: 230,
        delay: index * 150,
        easing: "cubic-bezier(.2, .72, .25, 1)",
        fill: "forwards",
      });
      animation.onfinish = () => {
        ghost.remove();
        ui.pieceMoveGhosts = ui.pieceMoveGhosts.filter((item) => item !== ghost);
        if (destination) destination.style.visibility = "";
      };
    }
    let sound = isCastle ? "castle" : isPromotion ? "promote" : captured ? "capture" : "move";
    if (index === moves.length - 1) {
      if (currentState?.result?.termination === "checkmate") sound = null;
      else if (currentState?.in_check) sound = "check";
    }
    if (sound && !(index === 0 && ui.suppressNextMoveSound)) {
      if (index === 0) playMoveSound(sound);
      else window.setTimeout(() => playMoveSound(sound), index * 200);
    }

    if (movingPiece) {
      delete pieces[move.from];
      if (isEnPassant) delete pieces[`${move.to[0]}${move.from[1]}`];
      pieces[move.to] = move.promotion ? { ...movingPiece, type: move.promotion } : movingPiece;
      if (movingPiece.type === "k" && Math.abs(FILES.indexOf(move.from[0]) - FILES.indexOf(move.to[0])) === 2) {
        const rank = move.from[1];
        const rookFrom = move.to[0] > move.from[0] ? `h${rank}` : `a${rank}`;
        const rookTo = move.to[0] > move.from[0] ? `f${rank}` : `d${rank}`;
        if (pieces[rookFrom]) {
          pieces[rookTo] = pieces[rookFrom];
          delete pieces[rookFrom];
        }
      }
    }
  });
}

function renderBoard(animationMoves = [], previousState = null) {
  const state = ui.studyGame?.frames[ui.studyGame.cursor] || ui.state || initialPreview();
  const board = $("#chessboard");
  const isBlackView = ui.flipped ? ui.playerColor !== "black" : ui.playerColor === "black";
  const rankOrder = isBlackView ? "12345678" : RANKS;
  const fileOrder = isBlackView ? "hgfedcba" : FILES;
  const files = [...fileOrder];
  const ranks = [...rankOrder];
  const legalFromSelected = (state.legal_moves || []).filter((move) => move.slice(0, 2) === ui.selected);
  const queueingPremove = Boolean(ui.engineThinking && state.started && !state.result && !ui.busy);
  const targets = queueingPremove ? premoveTargets(ui.selected, state) : legalFromSelected.map((move) => move.slice(2, 4));
  const legalTargets = new Set(targets);
  const lastMove = state.last_move || null;
  const currentColor = state.to_move || "white";
  const canMove = Boolean(!ui.studyGame && state.started && !state.result && !ui.busy && !ui.animating && !ui.engineThinking && currentColor === ui.playerColor && ui.model.available);
  const canQueuePremove = Boolean(!ui.studyGame && state.started && !state.result && !ui.busy && ui.engineThinking && currentColor !== ui.playerColor && ui.model.available);
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
      if (isTarget && queueingPremove) classes.push("premove-target");
      if (ui.premove?.from === square) classes.push("premove-source");
      if (ui.premove?.to === square) classes.push("premove-destination");
      if (state.in_check === square) classes.push("in-check");
      const pieceName = piece ? PIECE_NAMES[piece.type] : "";
      const pieceGlyph = piece ? `<img class="piece-svg piece-${piece.color}" src="/assets/pieces/cburnett/${piece.color}_${pieceName}.svg" alt="" draggable="false" />` : "";
      const description = piece ? `${piece.color} ${piece.type} on ${square}` : square;
      squares.push(`<button class="${classes.join(" ")}" type="button" role="gridcell" data-square="${square}" aria-label="${description}" aria-selected="${ui.selected === square}">${pieceGlyph}</button>`);
    });
  });

  ui.pieceMoveGhosts.forEach((ghost) => ghost.remove());
  ui.pieceMoveGhosts = [];
  board.querySelectorAll(".piece-svg").forEach((piece) => { piece.style.visibility = ""; });
  board.innerHTML = squares.join("");
  if (ui.pendingDragGhost) {
    board.querySelector(`[data-square="${ui.pendingDragGhost.from}"] .piece-svg`)?.style.setProperty("visibility", "hidden");
  }
  board.classList.toggle("is-disabled-game", !canMove && !canQueuePremove);
  $("#rank-labels").innerHTML = ranks.map((rank) => {
    const shade = (FILES.indexOf(files[0]) + Number(rank)) % 2 === 0 ? "coordinate-light" : "coordinate-dark";
    return `<span class="${shade}">${rank}</span>`;
  }).join("");
  $("#file-labels").innerHTML = files.map((file) => {
    const shade = (FILES.indexOf(file) + Number(ranks[ranks.length - 1])) % 2 === 0 ? "coordinate-light" : "coordinate-dark";
    return `<span class="${shade}">${file}</span>`;
  }).join("");
  board.querySelectorAll(".square").forEach((squareButton) => {
    squareButton.addEventListener("click", (event) => {
      if (ui.suppressBoardClick) {
        ui.suppressBoardClick = false;
        event.preventDefault();
        return;
      }
      onSquareClick(squareButton.dataset.square);
    });
    squareButton.addEventListener("pointerdown", (event) => {
      const source = squareButton.dataset.square;
      const selectedPiece = state.pieces?.[source];
      if (event.button !== 0 || (!canMove && !canQueuePremove) || selectedPiece?.color !== ui.playerColor) return;
      event.preventDefault();
      const pieceImage = squareButton.querySelector(".piece-svg");
      if (!pieceImage) return;
      const rect = pieceImage.getBoundingClientRect();
      ui.drag = {
        pointerId: event.pointerId,
        source,
        startX: event.clientX,
        startY: event.clientY,
        offsetX: event.clientX - rect.left,
        offsetY: event.clientY - rect.top,
        image: pieceImage,
        clone: null,
        moved: false,
      };
      squareButton.setPointerCapture(event.pointerId);
    });
    squareButton.addEventListener("pointermove", (event) => {
      const drag = ui.drag;
      if (!drag || drag.pointerId !== event.pointerId) return;
      if (!drag.moved && Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY) > 4) {
        drag.moved = true;
        ui.selected = drag.source;
        drag.clone = drag.image.cloneNode(true);
        drag.clone.classList.add("piece-drag-ghost");
        drag.clone.style.width = `${drag.image.getBoundingClientRect().width}px`;
        drag.clone.style.height = `${drag.image.getBoundingClientRect().height}px`;
        document.body.append(drag.clone);
        drag.image.classList.add("is-drag-source");
        board.classList.add("is-dragging");
      }
      if (drag.moved) {
        event.preventDefault();
        drag.currentX = event.clientX - drag.offsetX;
        drag.currentY = event.clientY - drag.offsetY;
        drag.clone.style.transform = `translate3d(${drag.currentX}px, ${drag.currentY}px, 0)`;
      }
    });
    squareButton.addEventListener("pointerup", (event) => {
      const drag = ui.drag;
      if (!drag || drag.pointerId !== event.pointerId) return;
      ui.drag = null;
      if (drag.moved) {
        ui.suppressBoardClick = true;
        const target = document.elementFromPoint(event.clientX, event.clientY)?.closest(".square[data-square]")?.dataset.square;
        drag.image.classList.remove("is-drag-source");
        board.classList.remove("is-dragging");
        if (target && canMove) {
          const destination = board.querySelector(`[data-square="${target}"]`);
          const rect = destination?.getBoundingClientRect();
          if (drag.clone && rect) {
            const width = drag.clone.getBoundingClientRect().width;
            const height = drag.clone.getBoundingClientRect().height;
            drag.clone.classList.add("piece-pending-ghost");
            drag.clone.style.transform = `translate3d(${rect.left + (rect.width - width) / 2}px, ${rect.top + (rect.height - height) / 2}px, 0)`;
            ui.pendingDragGhost = { clone: drag.clone, from: drag.source };
          } else {
            drag.clone?.remove();
          }
          submitMove(drag.source, target);
        } else if (target && canQueuePremove && premoveTargets(drag.source, state).includes(target)) {
          drag.clone?.remove();
          queuePremove(drag.source, target);
        } else {
          drag.clone?.remove();
          renderBoard();
        }
        window.setTimeout(() => { ui.suppressBoardClick = false; }, 0);
      } else {
        // The resulting click event keeps click-to-select behavior in one path.
      }
    });
  });
  animateBoardMoves(animationMoves, previousState, state);
}

function onSquareClick(square) {
  const isPlayersTurn = ui.state?.to_move === ui.playerColor;
  if (ui.studyGame || ui.busy || !ui.model.available || !ui.state?.started || ui.state.result || (!isPlayersTurn && !ui.engineThinking)) return;
  unlockAudio();
  const piece = ui.state.pieces?.[square];
  if (!ui.selected) {
    if (piece?.color === ui.playerColor) {
      ui.selected = square;
      ui.premove = null;
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
    ui.premove = null;
    renderBoard();
    return;
  }
  if (!isPlayersTurn && ui.engineThinking) {
    if (premoveTargets(ui.selected, ui.state).includes(square)) queuePremove(ui.selected, square);
    else {
      ui.selected = null;
      renderBoard();
    }
    return;
  }
  const move = (ui.state.legal_moves || []).find((candidate) => candidate.slice(0, 2) === ui.selected && candidate.slice(2, 4) === square);
  if (move) submitMove(ui.selected, square, move[4] || "q");
  else {
    ui.selected = null;
    renderBoard();
  }
}

function queuePremove(from, to) {
  if (!ui.engineThinking || !premoveTargets(from, ui.state).includes(to)) return;
  ui.premove = { from, to };
  ui.selected = null;
  setTurnMessage("Premove queued", `Whisker will try ${from}–${to} on your turn.`, true);
  renderBoard();
}

function playQueuedPremove() {
  if (!ui.premove || !ui.state?.started || ui.state.result || ui.state.to_move !== ui.playerColor || ui.engineThinking || ui.busy) return;
  const { from, to } = ui.premove;
  const candidate = ui.state.legal_moves?.find((move) => move.slice(0, 2) === from && move.slice(2, 4) === to);
  ui.premove = null;
  if (candidate) submitMove(from, to, candidate[4] || "q");
  else {
    setTurnMessage("Premove unavailable", "The position changed, so that move is no longer legal.");
    showToast("Premove was no longer legal.");
    renderBoard();
  }
}

async function submitMove(from, to, promotion = "q") {
  if (ui.busy || !ui.state?.started) return;
  const candidate = ui.state.legal_moves?.find((move) => move.slice(0, 2) === from && move.slice(2, 4) === to);
  if (!candidate) {
    ui.pendingDragGhost?.clone.remove();
    ui.pendingDragGhost = null;
    ui.selected = null;
    renderBoard();
    return;
  }
  const movingPiece = ui.state.pieces?.[from];
  const captured = Boolean(ui.state.pieces?.[to])
    || (movingPiece?.type === "p" && from[0] !== to[0]);
  const castle = movingPiece?.type === "k" && Math.abs(FILES.indexOf(from[0]) - FILES.indexOf(to[0])) === 2;
  const sound = castle ? "castle" : candidate.length === 5 ? "promote" : captured ? "capture" : "move";
  ui.suppressNextMoveSound = true;
  playMoveSound(sound);
  ui.pendingMove = { from, to, promotion: candidate.length === 5 ? promotion : null };
  unlockAudio();
  ui.busy = true;
  ui.selected = null;
  updateGameControls();
  renderBoard();
  setTurnMessage("Whisker is thinking…", "Your local chess companion is choosing a move.", true);
  try {
    const response = await api("/api/move", { from, to, promotion: candidate.length === 5 ? promotion : null });
    ui.busy = false;
    acceptState(response.state);
    if (ui.engineThinking && !ui.animating) void requestEngineMove();
  } catch (error) {
    ui.pendingMove = null;
    ui.suppressNextMoveSound = false;
    ui.pendingDragGhost?.clone.remove();
    ui.pendingDragGhost = null;
    ui.busy = false;
    showToast(error.message || "That move could not be made.");
    const refreshed = await refreshState();
    if (!refreshed) renderBoard();
  } finally {
    ui.busy = false;
    updateGameControls();
  }
}

async function requestEngineMove() {
  if (!ui.engineThinking || ui.busy) return;
  try {
    await new Promise((resolve) => window.setTimeout(resolve, PREMOVE_WINDOW_MS));
    if (!ui.engineThinking || !ui.state?.started || ui.state.result) return;
    const response = await api("/api/engine-move", {});
    acceptState(response.state);
  } catch (error) {
    ui.engineThinking = false;
    showToast(error.message === "Not found"
      ? "Restart the Whisker server to enable premoves."
      : error.message || "Whisker could not finish its move.");
    renderBoard();
    updateGameControls();
  }
}

function renderMoveHistory(history = [], activePly = history.length - 1, interactive = false) {
  const target = $("#move-list");
  $("#move-count").textContent = `${history.length} ${history.length === 1 ? "move" : "moves"}`;
  if (!history.length) {
    target.innerHTML = `<div class="empty-moves"><p>No moves yet.<br /><small>Notation will appear here.</small></p></div>`;
    return;
  }
  const byNumber = new Map();
  history.forEach((entry, index) => {
    if (!byNumber.has(entry.number)) byNumber.set(entry.number, { number: entry.number });
    byNumber.get(entry.number)[entry.color] = entry.san;
    byNumber.get(entry.number)[`${entry.color}Index`] = index;
  });
  const moveCell = (san, index) => {
    if (!san) return '<span class="move-san"></span>';
    const active = index === activePly;
    const classes = `move-san ${active ? "latest is-active" : ""}`;
    return interactive
      ? `<button class="${classes} move-jump" type="button" data-ply="${index}" aria-current="${active ? "step" : "false"}">${san}</button>`
      : `<span class="${classes}">${san}</span>`;
  };
  target.innerHTML = [...byNumber.values()].map((row) => `
    <div class="move-row"><span class="move-number">${row.number}.</span>${moveCell(row.white, row.whiteIndex)}${moveCell(row.black, row.blackIndex)}</div>
  `).join("");
  if (interactive) {
    target.querySelectorAll(".move-jump").forEach((button) => {
      button.addEventListener("click", () => setStudyPly(Number(button.dataset.ply) + 1));
    });
    target.querySelector(`[data-ply="${activePly}"]`)?.scrollIntoView({ block: "nearest" });
  } else {
    target.scrollTop = target.scrollHeight;
  }
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
  $("#turn-icon").textContent = over ? "✓" : live ? "●" : "○";
}

function acceptState(state) {
  if (!state) return;
  ui.moveTimeouts.forEach((timeout) => window.clearTimeout(timeout));
  ui.moveTimeouts = [];
  if (!state.pieces && !state.started) state = initialPreview();
  const previousState = ui.state;
  const checkmateJustOccurred = state.result?.termination === "checkmate"
    && previousState?.result?.termination !== "checkmate";
  const previousMoveCount = previousState?.history?.length || 0;
  const moveCount = state.history?.length || 0;
  const addedMoves = moveCount - previousMoveCount;
  const animationMoves = [];
  let animationSourceState = previousState;
  if (state.started && ui.pendingMove && addedMoves > 0) {
    if (ui.pendingDragGhost) {
      ui.pendingDragGhost.clone.remove();
      ui.pendingDragGhost = null;
    } else {
      animationMoves.push(ui.pendingMove);
    }
    const engineMove = parseUciMove(state.last_move);
    if (addedMoves > 1 && engineMove && `${engineMove.from}${engineMove.to}` !== `${ui.pendingMove.from}${ui.pendingMove.to}`) {
      animationMoves.push(engineMove);
    }
    ui.pendingMove = null;
  } else if (state.started && ui.startingGame && moveCount > 0) {
    const firstEngineMove = parseUciMove(state.last_move);
    if (firstEngineMove) {
      animationMoves.push(firstEngineMove);
      animationSourceState = initialPreview();
    }
  } else if (state.started && addedMoves > 0 && state.last_move && state.last_move !== previousState?.last_move) {
    const latestMove = parseUciMove(state.last_move);
    if (latestMove) animationMoves.push(latestMove);
  }
  if (ui.animationTimer) window.clearTimeout(ui.animationTimer);
  ui.animationTimer = null;
  ui.animating = false;
  const reducedMotion = Boolean(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  if (animationMoves.length && !reducedMotion) {
    ui.animating = true;
    ui.animationTimer = window.setTimeout(() => {
      ui.animating = false;
      ui.animationTimer = null;
      renderBoard();
      updateGameControls();
      if (ui.engineThinking) void requestEngineMove();
      playQueuedPremove();
    }, 280 + Math.max(0, animationMoves.length - 1) * 150);
  }
  if (state.started) ui.startingGame = false;
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
  ui.engineThinking = Boolean(state.started && !state.result && state.to_move !== ui.playerColor);
  setAvatarPieces(ui.playerColor === "white" ? "black" : "white", ui.playerColor);
  if (!state.started) ui.flipped = false;
  $("#player-caption").textContent = `Playing as ${capitalize(ui.playerColor)}`;
  $("#game-title").textContent = state.mode ? `${capitalize(state.mode)} · ${state.time_control}` : "New game";
  if (!state.started) {
    setTurnMessage(ui.model.available ? "Board is ready" : "Waiting for your model", ui.model.available ? "Choose a time and start a game." : "Connect a checkpoint to begin playing.");
  } else if (state.result) {
    const won = state.result.winner === ui.playerColor;
    const draw = !state.result.winner;
    setTurnMessage(draw ? "A well-played draw" : won ? "A lovely win" : "Game over", state.result.message || (draw ? "The game is a draw." : won ? "Nicely done." : "There is always another game."), false, true);
  } else if (ui.busy || ui.engineThinking || state.to_move !== ui.playerColor) {
    setTurnMessage("Whisker to move", ui.engineThinking
      ? "Click one of your pieces, then its destination to queue a premove."
      : "Your local chess companion is considering the position.", true);
  } else {
    setTurnMessage("Your move", "Take your time. The clock is yours.", true);
  }
  const engineBadge = $("#engine-turn-badge");
  if (engineBadge) {
    engineBadge.textContent = state.started
      ? state.result
        ? "FINISHED"
        : state.to_move === ui.playerColor
          ? "YOUR TURN"
          : "THINKING"
      : "READY";
    engineBadge.classList.toggle("is-live", state.started && !state.result);
  }
  $("#resign-button").disabled = !state.started || Boolean(state.result) || ui.busy || ui.animating;
  document.querySelectorAll(".side-option").forEach((option) => {
    const selected = option.dataset.color === ui.colorChoice;
    option.classList.toggle("is-selected", selected);
    option.setAttribute("aria-pressed", String(selected));
    option.disabled = Boolean(state.started && !state.result);
  });
  if (ui.studyGame) {
    renderModeControls();
    renderStudyView();
    return;
  }
  renderMoveHistory(state.history || []);
  renderModeControls();
  renderBoard(animationMoves, animationSourceState);
  ui.suppressNextMoveSound = false;
  if (checkmateJustOccurred) playMoveSound("gameend");
  renderClocks();
  updateGameControls();
  if (!ui.animating) playQueuedPremove();
}

function updateGameControls() {
  const button = $("#new-game-button");
  button.disabled = !ui.model.available || ui.busy || ui.animating || ui.engineThinking;
  button.classList.toggle("is-loading", ui.busy);
  $("#resign-button").disabled = !ui.state?.started || Boolean(ui.state?.result) || ui.busy || ui.animating;
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
    ? `Connected checkpoint: ${ui.model.checkpoint || "Whisker"} (${ui.model.device || "local device"}).`
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
    if (data.state) {
      acceptState(data.state);
      return true;
    }
    if (!ui.state) acceptState(initialPreview());
    return false;
  } catch {
    if (!ui.state) acceptState(initialPreview());
    return false;
  }
}

async function startGame() {
  if (!ui.model.available || ui.busy) return;
  const playerColor = ui.colorChoice === "random"
    ? (Math.random() < 0.5 ? "white" : "black")
    : ui.colorChoice;
  ui.playerColor = playerColor;
  unlockAudio();
  ui.startingGame = true;
  ui.busy = true;
  updateGameControls();
  setTurnMessage("Setting the board…", "Getting a fresh game ready for you.", true);
  try {
    const data = await api("/api/new", {
      mode: ui.mode,
      minutes: ui.control.minutes,
      increment: ui.control.increment,
      player_color: playerColor,
    });
    ui.busy = false;
    if (ui.studyGame) clearStudyView();
    ui.flipped = false;
    acceptState(data.state);
    if (ui.engineThinking && !ui.animating) void requestEngineMove();
    showToast(`${capitalize(ui.mode)} game started.`);
  } catch (error) {
    ui.busy = false;
    ui.startingGame = false;
    setTurnMessage("Could not start a game", error.message || "Check your engine setup.", false, true);
    showToast(error.message || "Could not start a game.");
    renderBoard();
  } finally {
    ui.busy = false;
    updateGameControls();
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

function looksLikePgn(text) {
  if (typeof text !== "string" || !text.trim()) return false;
  return /^\s*\[[A-Za-z][\w-]*\s+"/m.test(text)
    || /(?:^|\s)1\.\s*(?:\.\.\s*)?(?:[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8]|O-O(?:-O)?)/m.test(text);
}

function openPgnDialog(text = "") {
  const textarea = $("#pgn-text");
  if (text) textarea.value = text;
  const dialog = $("#pgn-dialog");
  if (!dialog.open) dialog.showModal();
  window.requestAnimationFrame(() => {
    textarea.focus({ preventScroll: true });
    textarea.setSelectionRange(textarea.value.length, textarea.value.length);
  });
}

function stopStudyPlayback() {
  if (ui.studyTimer) window.clearInterval(ui.studyTimer);
  ui.studyTimer = null;
}

function setStudyPly(ply) {
  if (!ui.studyGame) return;
  unlockAudio();
  const previousPly = ui.studyGame.cursor;
  const lastPly = ui.studyGame.frames.length - 1;
  ui.studyGame.cursor = Math.max(0, Math.min(lastPly, Number(ply) || 0));
  if (ui.studyGame.cursor === lastPly) stopStudyPlayback();
  const animationMoves = [];
  let animationSource = null;
  if (ui.studyGame.cursor === previousPly + 1) {
    const frame = ui.studyGame.frames[ui.studyGame.cursor];
    const move = parseUciMove(frame.last_move);
    if (move) {
      animationMoves.push(move);
      animationSource = ui.studyGame.frames[previousPly];
    }
  } else if (ui.studyGame.cursor === previousPly - 1) {
    const move = parseUciMove(ui.studyGame.frames[previousPly].last_move);
    if (move) {
      animationMoves.push({ ...move, from: move.to, to: move.from });
      animationSource = ui.studyGame.frames[previousPly];
    }
  }
  renderStudyView(animationMoves, animationSource);
}

function renderStudyView(animationMoves = [], animationSource = null) {
  const study = ui.studyGame;
  if (!study) return;
  const { headers, history, frames } = study;
  const frame = frames[study.cursor];
  const white = headers.White && headers.White !== "?" ? headers.White : "White";
  const black = headers.Black && headers.Black !== "?" ? headers.Black : "Black";
  const blackView = ui.flipped ? ui.playerColor !== "black" : ui.playerColor === "black";
  const topColor = blackView ? "white" : "black";
  const bottomColor = topColor === "white" ? "black" : "white";
  setAvatarPieces(topColor, bottomColor);
  const names = { white, black };

  document.body.classList.add("is-reviewing");
  $("#review-controls").hidden = false;
  $("#resign-button").hidden = true;
  $(".opponent-strip .person-name").textContent = names[topColor];
  $(".self-strip .person-name").textContent = names[bottomColor];
  $("#engine-caption").textContent = `${capitalize(topColor)} · PGN review`;
  $("#player-caption").textContent = `${capitalize(bottomColor)} · PGN review`;
  $("#game-title").textContent = `${white} vs ${black}`;

  const eventParts = [headers.Event, headers.Date].filter((value) => value && value !== "?");
  const currentMove = history[study.cursor - 1];
  const positionText = study.cursor === 0
    ? "Starting position"
    : study.cursor === history.length
      ? `Final position${headers.Result && headers.Result !== "*" ? ` · ${headers.Result}` : ""}`
      : `${currentMove.number}${currentMove.color === "black" ? "..." : "."} ${currentMove.san}`;
  $("#review-position").textContent = positionText;
  $("#review-range").max = String(history.length);
  $("#review-range").value = String(study.cursor);
  const playButton = $("#review-play");
  playButton.textContent = ui.studyTimer ? "Ⅱ" : "▶";
  playButton.setAttribute("aria-label", ui.studyTimer ? "Pause game" : "Play game");
  playButton.title = ui.studyTimer ? "Pause game" : "Play game";
  $("#review-first").disabled = study.cursor === 0;
  $("#review-previous").disabled = study.cursor === 0;
  $("#review-next").disabled = study.cursor === history.length;
  $("#review-last").disabled = study.cursor === history.length;

  const turnTitle = study.cursor === history.length ? "Final position" : "Game review";
  const detail = `${positionText} · ${study.cursor} of ${history.length} plies`;
  setTurnMessage(turnTitle, [...eventParts, detail].join(" · "));
  const engineBadge = $("#engine-turn-badge");
  if (engineBadge) {
    engineBadge.textContent = "REVIEW";
    engineBadge.classList.remove("is-live");
  }
  renderMoveHistory(history, study.cursor - 1, true);
  renderBoard(animationMoves, animationSource);
  renderClocks();
  updateGameControls();
}

function restorePlayerLabels() {
  const opponentName = $(".opponent-strip .person-name");
  opponentName.replaceChildren(document.createTextNode("Whisker "));
  const statusDot = document.createElement("span");
  statusDot.className = `online-indicator ${ui.model.available ? "is-online" : "is-offline"}`;
  statusDot.id = "engine-status-dot";
  opponentName.append(statusDot);

  const playerName = $(".self-strip .person-name");
  playerName.replaceChildren(document.createTextNode("You"));
  $("#engine-caption").textContent = ui.model.available ? "Your local policy model" : "Waiting for a model checkpoint";
  $("#player-caption").textContent = `Playing as ${capitalize(ui.playerColor)}`;
}

function clearStudyView() {
  stopStudyPlayback();
  ui.studyGame = null;
  document.body.classList.remove("is-reviewing");
  $("#review-controls").hidden = true;
  $("#resign-button").hidden = false;
  restorePlayerLabels();
}

function exitStudy() {
  if (!ui.studyGame) return;
  clearStudyView();
  if (ui.state) acceptState(ui.state);
}

async function reviewPgn() {
  const pgn = $("#pgn-text").value.trim();
  const status = $("#pgn-status");
  const button = $("#review-pgn-button");
  if (!pgn) {
    status.textContent = "Paste a PGN or open a .pgn file first.";
    return;
  }
  if (ui.state?.started && !ui.state.result) {
    status.textContent = "Finish the active game before opening a PGN review.";
    return;
  }

  button.disabled = true;
  button.textContent = "Reading…";
  status.textContent = "Reading the game…";
  try {
    const response = await api("/api/pgn", { pgn });
    const study = response.study;
    if (!study?.frames?.length || !study.history?.length) throw new Error("No moves were found in that PGN.");
    clearStudyView();
    ui.studyGame = { ...study, cursor: 0 };
    ui.flipped = false;
    $("#pgn-dialog").close();
    renderStudyView();
    showToast(`Loaded ${study.history.length} plies for review.`);
  } catch (error) {
    status.textContent = error.message || "Could not read that PGN.";
  } finally {
    button.disabled = false;
    button.textContent = "Review game";
  }
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
      ui.colorChoice = button.dataset.color;
      if (ui.colorChoice !== "random") ui.playerColor = ui.colorChoice;
      document.querySelectorAll(".side-option").forEach((option) => {
        const selected = option === button;
        option.classList.toggle("is-selected", selected);
        option.setAttribute("aria-pressed", String(selected));
      });
      if (ui.state && !ui.state.started && ui.colorChoice !== "random") ui.state.player_color = ui.playerColor;
      $("#player-caption").textContent = ui.colorChoice === "random"
        ? "Side chosen at random"
        : `Playing as ${capitalize(ui.playerColor)}`;
      if (ui.studyGame) renderStudyView();
      else renderBoard();
    });
  });
  $("#new-game-button").addEventListener("click", startGame);
  $("#flip-board-button").addEventListener("click", () => {
    ui.flipped = !ui.flipped;
    if (ui.studyGame) renderStudyView();
    else renderBoard();
  });
  $("#resign-button").addEventListener("click", resignGame);
  const pgnDialog = $("#pgn-dialog");
  const pgnInput = $("#pgn-file-input");
  $("#open-pgn-button").addEventListener("click", () => openPgnDialog());
  $("#close-pgn-dialog").addEventListener("click", () => pgnDialog.close());
  $("#choose-pgn-file").addEventListener("click", () => pgnInput.click());
  $("#review-pgn-button").addEventListener("click", reviewPgn);
  pgnInput.addEventListener("change", async () => {
    const [file] = pgnInput.files || [];
    if (file) {
      try {
        $("#pgn-text").value = await file.text();
        $("#pgn-status").textContent = `${file.name} ready to review.`;
      } catch {
        $("#pgn-status").textContent = "Could not read that PGN file.";
      }
    }
    pgnInput.value = "";
  });
  $("#review-first").addEventListener("click", () => setStudyPly(0));
  $("#review-previous").addEventListener("click", () => setStudyPly((ui.studyGame?.cursor || 0) - 1));
  $("#review-next").addEventListener("click", () => setStudyPly((ui.studyGame?.cursor || 0) + 1));
  $("#review-last").addEventListener("click", () => setStudyPly((ui.studyGame?.frames.length || 1) - 1));
  $("#review-range").addEventListener("input", (event) => setStudyPly(event.currentTarget.value));
  $("#review-play").addEventListener("click", () => {
    if (!ui.studyGame) return;
    unlockAudio();
    if (ui.studyTimer) {
      stopStudyPlayback();
      renderStudyView();
      return;
    }
    if (ui.studyGame.cursor >= ui.studyGame.frames.length - 1) ui.studyGame.cursor = 0;
    renderStudyView();
    ui.studyTimer = window.setInterval(() => setStudyPly(ui.studyGame.cursor + 1), 800);
    renderStudyView();
  });
  $("#exit-review-button").addEventListener("click", exitStudy);
  document.addEventListener("paste", (event) => {
    const target = event.target;
    if (target instanceof HTMLElement && target.closest("input, textarea, [contenteditable='true']")) return;
    const text = event.clipboardData?.getData("text/plain") || "";
    if (!looksLikePgn(text)) return;
    event.preventDefault();
    openPgnDialog(text);
  });
  $("#sound-toggle").addEventListener("click", (event) => {
    ui.soundEnabled = !ui.soundEnabled;
    event.currentTarget.setAttribute("aria-pressed", String(ui.soundEnabled));
    event.currentTarget.title = ui.soundEnabled ? "Turn move sounds off" : "Turn move sounds on";
    $("#sound-label").textContent = ui.soundEnabled ? "Sound on" : "Sound off";
    event.currentTarget.classList.toggle("is-muted", !ui.soundEnabled);
    if (ui.soundEnabled) {
      void unlockAudio();
    } else {
      ui.audioPlayers = null;
    }
    showToast(ui.soundEnabled ? "Move sounds are on." : "Move sounds are off.");
  });
  $("#engine-help-button").addEventListener("click", () => $("#setup-dialog").showModal());
  $("#engine-help-top").addEventListener("click", () => $("#setup-dialog").showModal());
  $("#theme-button").addEventListener("click", () => {
    const enabled = document.body.classList.toggle("dark-theme");
    const button = $("#theme-button");
    button.setAttribute("aria-pressed", String(enabled));
    button.title = enabled ? "Switch to light theme" : "Switch to dark theme";
    button.setAttribute("aria-label", button.title);
    try { localStorage.setItem("whisker-theme", enabled ? "dark" : "light"); } catch { /* Theme still works for this session. */ }
    showToast(enabled ? "Dark theme enabled." : "Light theme enabled.");
  });
}

async function initialize() {
  try {
    const darkTheme = localStorage.getItem("whisker-theme") === "dark";
    document.body.classList.toggle("dark-theme", darkTheme);
    const themeButton = $("#theme-button");
    themeButton.setAttribute("aria-pressed", String(darkTheme));
    themeButton.title = darkTheme ? "Switch to light theme" : "Switch to dark theme";
    themeButton.setAttribute("aria-label", themeButton.title);
  } catch { /* Use the default light theme when storage is unavailable. */ }
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
      detail: "Start Whisker with python -m src.serve to connect the local engine.",
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
    if (ui.engineThinking || ui.animating) return;
    if (ui.state.clocks_ms[clockKey] <= 0 && !syncingFlag) {
      syncingFlag = true;
      refreshState().finally(() => { syncingFlag = false; });
    }
  }, 200);
}

initialize();
