/**
 * TradeMind Phase 9.5B Client-Side Integration Script
 * Connects Frontend UI to:
 * - /api/health & /api/v1/system/status (System Diagnostics)
 * - /api/v1/assets & /api/v1/assets/refresh (Historical Market Data)
 * - /api/v1/models (PPO Model Registry & Policy Management)
 * - /api/v1/trading/portfolio, /step, /trades, /risk-events (Simulation & Risk Gate)
 */

document.addEventListener("DOMContentLoaded", () => {
    // -------------------------------------------------------------------------
    // DOM Elements
    // -------------------------------------------------------------------------
    const btnCheckHealth = document.getElementById("btn-check-health");
    const statApiStatus = document.getElementById("stat-api-status");
    const statDbStatus = document.getElementById("stat-db-status");
    const statModelsCount = document.getElementById("stat-models-count");
    const statLatency = document.getElementById("stat-latency");
    const jsonOutput = document.getElementById("health-json-output");
    const payloadTime = document.getElementById("payload-time");
    const headerStatusText = document.getElementById("header-status-text");
    const headerPulseDot = document.getElementById("header-pulse-dot");

    // Datasets
    const btnRefreshAllData = document.getElementById("btn-refresh-all-data");
    const datasetsTableBody = document.getElementById("datasets-table-body");
    const dataStatusMessage = document.getElementById("data-status-message");

    // Models
    const btnSyncModels = document.getElementById("btn-sync-models");
    const modelsTableBody = document.getElementById("models-table-body");
    const modelStatusMessage = document.getElementById("model-status-message");

    // Simulation
    const selectSimSymbol = document.getElementById("select-sim-symbol");
    const btnRunSimStep = document.getElementById("btn-run-sim-step");
    const simSpinner = document.getElementById("sim-spinner");
    const simCash = document.getElementById("sim-cash");
    const simTotalValue = document.getElementById("sim-total-value");
    const simHolding = document.getElementById("sim-holding");
    const simPeakDd = document.getElementById("sim-peak-dd");
    const stepOutcomeCard = document.getElementById("step-outcome-card");
    const outcomePpoAction = document.getElementById("outcome-ppo-action");
    const outcomeRiskStatus = document.getElementById("outcome-risk-status");
    const outcomeEffectiveAction = document.getElementById("outcome-effective-action");
    const outcomeRegime = document.getElementById("outcome-regime");
    const outcomeDetails = document.getElementById("outcome-details");
    const tradesTableBody = document.getElementById("trades-table-body");
    const riskEventsTableBody = document.getElementById("risk-events-table-body");

    // -------------------------------------------------------------------------
    // Utility Helpers
    // -------------------------------------------------------------------------
    function formatCurrencyINR(amount) {
        if (amount === undefined || amount === null || isNaN(amount)) return "₹0.00";
        return "₹" + Number(amount).toLocaleString("en-IN", {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
        });
    }

    function showBanner(elem, message, isError = false) {
        if (!elem) return;
        elem.style.display = "flex";
        elem.className = isError ? "notice-banner" : "notice-banner info";
        elem.innerHTML = `<span>${isError ? "⚠️" : "ℹ️"}</span><div>${message}</div>`;
        setTimeout(() => {
            elem.style.display = "none";
        }, 8000);
    }

    // -------------------------------------------------------------------------
    // 1. System Health & Diagnostics
    // -------------------------------------------------------------------------
    async function fetchHealthStatus() {
        const startTime = performance.now();
        if (statApiStatus) statApiStatus.textContent = "Probing...";

        try {
            const [healthRes, statusRes] = await Promise.all([
                fetch("/api/health", { headers: { "Accept": "application/json" }, cache: "no-store" }),
                fetch("/api/v1/system/status", { headers: { "Accept": "application/json" }, cache: "no-store" }),
            ]);

            const latency = Math.round(performance.now() - startTime);
            const healthData = await healthRes.json();
            const statusData = statusRes.ok ? await statusRes.json() : null;

            if (healthRes.ok && healthData.status === "healthy") {
                if (statApiStatus) {
                    statApiStatus.textContent = "Healthy (200 OK)";
                    statApiStatus.className = "stat-value text-emerald";
                }
                if (statLatency) statLatency.textContent = `${latency} ms`;
                if (statDbStatus && statusData && statusData.data) {
                    statDbStatus.textContent = `${statusData.data.database_status} (SQLite)`;
                    statDbStatus.className = "stat-value text-emerald";
                }
                if (jsonOutput) {
                    jsonOutput.innerHTML = `<code>${JSON.stringify({ health: healthData, system_status: statusData ? statusData.data : null }, null, 2)}</code>`;
                }
                if (payloadTime) {
                    payloadTime.textContent = `Updated: ${new Date().toLocaleTimeString()}`;
                }
                if (headerStatusText) headerStatusText.textContent = "Online";
                if (headerPulseDot) {
                    headerPulseDot.style.backgroundColor = "var(--accent-emerald)";
                    headerPulseDot.style.boxShadow = "0 0 10px var(--accent-emerald)";
                }
            } else {
                throw new Error(`Unexpected health status: ${healthRes.status}`);
            }
        } catch (error) {
            const latency = Math.round(performance.now() - startTime);
            if (statApiStatus) {
                statApiStatus.textContent = "Degraded / Offline";
                statApiStatus.className = "stat-value text-rose";
            }
            if (statLatency) statLatency.textContent = `${latency} ms`;
            if (jsonOutput) {
                jsonOutput.innerHTML = `<code>{\n  "error": "${error.message}",\n  "status": "unreachable"\n}</code>`;
            }
            if (payloadTime) {
                payloadTime.textContent = `Failed: ${new Date().toLocaleTimeString()}`;
            }
            if (headerStatusText) headerStatusText.textContent = "Offline";
            if (headerPulseDot) {
                headerPulseDot.style.backgroundColor = "var(--accent-rose)";
                headerPulseDot.style.boxShadow = "0 0 10px var(--accent-rose)";
            }
        }
    }

    // -------------------------------------------------------------------------
    // 2. Historical Market Datasets
    // -------------------------------------------------------------------------
    async function loadDatasets() {
        if (!datasetsTableBody) return;
        try {
            const res = await fetch("/api/v1/assets", { cache: "no-store" });
            const result = await res.json();
            if (!res.ok || !result.success) {
                throw new Error(result.error || "Failed to load datasets.");
            }

            const assets = result.data || [];
            if (assets.length === 0) {
                datasetsTableBody.innerHTML = `
                    <tr>
                        <td colspan="7" class="empty-state">
                            <div class="empty-state-icon">📂</div>
                            <div class="empty-state-text">No configured assets found.</div>
                        </td>
                    </tr>`;
                return;
            }

            datasetsTableBody.innerHTML = assets.map(asset => {
                const isAvail = asset.data_available === true;
                const statusBadge = isAvail
                    ? `<span class="badge badge-success">Available</span>`
                    : `<span class="badge badge-danger">Missing</span>`;

                const dateRange = (asset.first_date && asset.last_date)
                    ? `${asset.first_date.split("T")[0]} → ${asset.last_date.split("T")[0]}`
                    : "Not Ingested";

                const checksum = asset.checksum
                    ? `<span class="mono-cell" title="${asset.checksum}">${asset.checksum.substring(0, 12)}...</span>`
                    : "<span class='text-muted'>--</span>";

                return `
                    <tr>
                        <td><strong>${asset.symbol}</strong></td>
                        <td>${asset.name || "--"}<br><small style="color:var(--text-muted);">${asset.sector || ""}</small></td>
                        <td class="mono-cell">${dateRange}</td>
                        <td class="mono-cell">${asset.total_bars !== undefined ? asset.total_bars.toLocaleString() : "--"}</td>
                        <td>${checksum}</td>
                        <td>${statusBadge}</td>
                        <td>
                            <button class="btn btn-secondary btn-sm btn-refresh-symbol" data-symbol="${asset.symbol}">
                                <span>🔄</span> Refresh
                            </button>
                        </td>
                    </tr>`;
            }).join("");

            // Attach refresh listeners to each symbol button
            document.querySelectorAll(".btn-refresh-symbol").forEach(btn => {
                btn.addEventListener("click", async (e) => {
                    const sym = btn.getAttribute("data-symbol");
                    await refreshDataset(sym, btn);
                });
            });

        } catch (err) {
            datasetsTableBody.innerHTML = `
                <tr>
                    <td colspan="7" class="empty-state text-rose">
                        <div class="empty-state-icon">⚠️</div>
                        <div class="empty-state-text">Error loading assets: ${err.message}</div>
                    </td>
                </tr>`;
        }
    }

    async function refreshDataset(symbol, btnElement = null) {
        if (btnElement) {
            btnElement.disabled = true;
            btnElement.innerHTML = `<span>⏳</span> Updating...`;
        }

        try {
            const res = await fetch("/api/v1/assets/refresh", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ symbol: symbol }),
            });
            const result = await res.json();
            if (res.ok && result.success) {
                showBanner(dataStatusMessage, `Dataset for <strong>${symbol}</strong> refreshed successfully! Merged rows: ${result.data.updated_total_rows || 'N/A'}`);
                await loadDatasets();
            } else {
                showBanner(dataStatusMessage, `Refresh failed for ${symbol}: ${result.error || "Unknown error"}`, true);
            }
        } catch (err) {
            showBanner(dataStatusMessage, `Network error refreshing ${symbol}: ${err.message}`, true);
        } finally {
            if (btnElement) {
                btnElement.disabled = false;
                btnElement.innerHTML = `<span>🔄</span> Refresh`;
            }
        }
    }

    if (btnRefreshAllData) {
        btnRefreshAllData.addEventListener("click", async () => {
            btnRefreshAllData.disabled = true;
            btnRefreshAllData.innerHTML = `<span>⏳</span> Refreshing All...`;
            try {
                const res = await fetch("/api/v1/assets/refresh", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ refresh_all: true }),
                });
                const result = await res.json();
                if (res.ok && result.success) {
                    showBanner(dataStatusMessage, `All datasets refreshed. Processed symbols: ${result.data.refreshed_symbols.length}`);
                    await loadDatasets();
                } else {
                    showBanner(dataStatusMessage, `Refresh all failed: ${result.error || "Unknown error"}`, true);
                }
            } catch (err) {
                showBanner(dataStatusMessage, `Network error refreshing all: ${err.message}`, true);
            } finally {
                btnRefreshAllData.disabled = false;
                btnRefreshAllData.innerHTML = `<span>🔄</span> Refresh All Configured`;
            }
        });
    }

    // -------------------------------------------------------------------------
    // 3. PPO Model Registry & Policy Lifecycle
    // -------------------------------------------------------------------------
    async function loadModels() {
        if (!modelsTableBody) return;
        try {
            const res = await fetch("/api/v1/models", { cache: "no-store" });
            const result = await res.json();
            if (!res.ok || !result.success) {
                throw new Error(result.error || "Failed to load models.");
            }

            const models = result.data || [];
            if (statModelsCount) {
                statModelsCount.textContent = `${models.length} Registered`;
            }

            if (models.length === 0) {
                modelsTableBody.innerHTML = `
                    <tr>
                        <td colspan="8" class="empty-state">
                            <div class="empty-state-icon">🧠</div>
                            <div class="empty-state-text">No model artifacts registered. Click 'Sync Local Checkpoints' to discover models.</div>
                        </td>
                    </tr>`;
                return;
            }

            modelsTableBody.innerHTML = models.map(m => {
                let statusBadge = `<span class="badge badge-neutral">${m.status}</span>`;
                if (m.status === "active") statusBadge = `<span class="badge badge-success">Active</span>`;
                else if (m.status === "approved") statusBadge = `<span class="badge badge-info">Approved</span>`;
                else if (m.status === "candidate") statusBadge = `<span class="badge badge-warning">Candidate</span>`;

                const activeBadge = m.is_active
                    ? `<span class="badge badge-success">✓ YES</span>`
                    : `<span class="badge badge-neutral">No</span>`;

                const checksum = m.artifact_checksum
                    ? `<span class="mono-cell" title="${m.artifact_checksum}">${m.artifact_checksum.substring(0, 10)}...</span>`
                    : "<span class='text-muted'>--</span>";

                let actionButtons = "";
                if (m.status === "candidate") {
                    actionButtons += `<button class="btn btn-outline btn-sm btn-approve-model" data-id="${m.id}" data-name="${m.model_id}">Approve</button> `;
                }
                if (m.status === "approved" && !m.is_active) {
                    actionButtons += `<button class="btn btn-success btn-sm btn-activate-model" data-id="${m.id}" data-name="${m.model_id}">Activate</button> `;
                }
                if (m.is_active) {
                    actionButtons += `<button class="btn btn-secondary btn-sm btn-rollback-model" data-symbol="${m.symbol}">Rollback</button>`;
                }

                return `
                    <tr>
                        <td><strong>${m.model_id}</strong><br><small style="color:var(--text-muted);">${m.name || ""}</small></td>
                        <td>${m.symbol || "PORTFOLIO"}</td>
                        <td>${m.algorithm}</td>
                        <td class="mono-cell">${m.obs_dimension}</td>
                        <td>${checksum}</td>
                        <td>${statusBadge}</td>
                        <td>${activeBadge}</td>
                        <td>${actionButtons || "<span class='text-muted'>None</span>"}</td>
                    </tr>`;
            }).join("");

            // Attach listeners to Approve buttons
            document.querySelectorAll(".btn-approve-model").forEach(btn => {
                btn.addEventListener("click", async () => {
                    const id = btn.getAttribute("data-id");
                    const name = btn.getAttribute("data-name");
                    await approveModel(id, name);
                });
            });

            // Attach listeners to Activate buttons
            document.querySelectorAll(".btn-activate-model").forEach(btn => {
                btn.addEventListener("click", async () => {
                    const id = btn.getAttribute("data-id");
                    const name = btn.getAttribute("data-name");
                    await activateModel(id, name);
                });
            });

            // Attach listeners to Rollback buttons
            document.querySelectorAll(".btn-rollback-model").forEach(btn => {
                btn.addEventListener("click", async () => {
                    const sym = btn.getAttribute("data-symbol");
                    await rollbackModel(sym);
                });
            });

        } catch (err) {
            modelsTableBody.innerHTML = `
                <tr>
                    <td colspan="8" class="empty-state text-rose">
                        <div class="empty-state-icon">⚠️</div>
                        <div class="empty-state-text">Error loading models: ${err.message}</div>
                    </td>
                </tr>`;
        }
    }

    async function approveModel(id, name) {
        try {
            const res = await fetch(`/api/v1/models/${id}/approve`, { method: "POST" });
            const result = await res.json();
            if (res.ok && result.success) {
                showBanner(modelStatusMessage, `Model <strong>${name}</strong> approved successfully!`);
                await loadModels();
            } else {
                showBanner(modelStatusMessage, `Approval failed: ${result.error || "Unknown error"}`, true);
            }
        } catch (err) {
            showBanner(modelStatusMessage, `Network error: ${err.message}`, true);
        }
    }

    async function activateModel(id, name) {
        try {
            const res = await fetch(`/api/v1/models/${id}/activate`, { method: "POST" });
            const result = await res.json();
            if (res.ok && result.success) {
                showBanner(modelStatusMessage, `Model <strong>${name}</strong> activated as active inference policy!`);
                await loadModels();
            } else {
                showBanner(modelStatusMessage, `Activation failed: ${result.error || "Unknown error"}`, true);
            }
        } catch (err) {
            showBanner(modelStatusMessage, `Network error: ${err.message}`, true);
        }
    }

    async function rollbackModel(symbol) {
        try {
            const res = await fetch(`/api/v1/models/${symbol}/rollback`, { method: "POST" });
            const result = await res.json();
            if (res.ok && result.success) {
                showBanner(modelStatusMessage, `Active model for <strong>${symbol}</strong> rolled back.`);
                await loadModels();
            } else {
                showBanner(modelStatusMessage, `Rollback failed: ${result.error || "Unknown error"}`, true);
            }
        } catch (err) {
            showBanner(modelStatusMessage, `Network error: ${err.message}`, true);
        }
    }

    if (btnSyncModels) {
        btnSyncModels.addEventListener("click", async () => {
            btnSyncModels.disabled = true;
            btnSyncModels.innerHTML = `<span>⏳</span> Syncing...`;
            try {
                const res = await fetch("/api/v1/models/sync", { method: "POST" });
                const result = await res.json();
                if (res.ok && result.success) {
                    showBanner(modelStatusMessage, `Synced checkpoints. New models registered: ${result.data.newly_registered_count}`);
                    await loadModels();
                } else {
                    showBanner(modelStatusMessage, `Sync failed: ${result.error || "Unknown error"}`, true);
                }
            } catch (err) {
                showBanner(modelStatusMessage, `Network error: ${err.message}`, true);
            } finally {
                btnSyncModels.disabled = false;
                btnSyncModels.innerHTML = `<span>🔍</span> Sync Local Checkpoints`;
            }
        });
    }

    // -------------------------------------------------------------------------
    // 4. Stock Bucket / Cart & Multi-Stock Portfolio Management
    // -------------------------------------------------------------------------
    const STOCK_COMPANIES = {
        "RELIANCE.NS": "Reliance Industries Limited",
        "TCS.NS": "Tata Consultancy Services",
        "INFY.NS": "Infosys Limited",
        "ICICIBANK.NS": "ICICI Bank Limited",
        "SBIN.NS": "State Bank of India",
    };

    let bucketItems = [
        { symbol: "RELIANCE.NS", weight: 0.35 },
        { symbol: "TCS.NS", weight: 0.35 },
        { symbol: "INFY.NS", weight: 0.30 },
    ];
    let bucketAllocationMode = "EQUAL_WEIGHT";
    let currentActivePortfolioName = "default";

    // Bucket DOM elements
    const bucketStockSelect = document.getElementById("bucket-stock-select");
    const btnAddToBucket = document.getElementById("btn-add-to-bucket");
    const radioModeEqual = document.getElementById("radio-mode-equal");
    const radioModeCustom = document.getElementById("radio-mode-custom");
    const btnEqualizeBucketWeights = document.getElementById("btn-equalize-bucket-weights");
    const btnClearBucket = document.getElementById("btn-clear-bucket");
    const bucketTableBody = document.getElementById("bucket-table-body");
    const bucketTotalWeightText = document.getElementById("bucket-total-weight-text");
    const bucketWeightBadge = document.getElementById("bucket-weight-badge");
    const inputPortfolioName = document.getElementById("input-portfolio-name");
    const inputPortfolioCapital = document.getElementById("input-portfolio-capital");
    const selectPortfolioStrategy = document.getElementById("select-portfolio-strategy");
    const selectPortfolioRisk = document.getElementById("select-portfolio-risk");
    const btnCreatePortfolio = document.getElementById("btn-create-portfolio");
    const createPortfolioMsg = document.getElementById("create-portfolio-msg");

    // Portfolio Dashboard DOM elements
    const selectActivePortfolio = document.getElementById("select-active-portfolio");
    const btnDeleteActivePortfolio = document.getElementById("btn-delete-active-portfolio");
    const btnRebalancePortfolio = document.getElementById("btn-rebalance-portfolio");
    const simTotalPnl = document.getElementById("sim-total-pnl");
    const holdingsTableBody = document.getElementById("holdings-table-body");
    const portfolioEquityChart = document.getElementById("portfolio-equity-chart");

    function updateBucketTotals() {
        const total = bucketItems.reduce((acc, it) => acc + (Number(it.weight) || 0), 0);
        const pct = (total * 100).toFixed(1);
        if (bucketTotalWeightText) {
            bucketTotalWeightText.textContent = `${pct}%`;
        }

        const isValid = Math.abs(total - 1.0) <= 0.015;
        if (bucketWeightBadge) {
            if (isValid) {
                bucketWeightBadge.className = "badge badge-success";
                bucketWeightBadge.textContent = "Valid (100%)";
                if (bucketTotalWeightText) bucketTotalWeightText.style.color = "var(--accent-emerald)";
            } else {
                bucketWeightBadge.className = "badge badge-danger";
                bucketWeightBadge.textContent = `Invalid (${pct}%)`;
                if (bucketTotalWeightText) bucketTotalWeightText.style.color = "var(--accent-rose)";
            }
        }
        return isValid;
    }

    function renderBucket() {
        if (!bucketTableBody) return;
        if (bucketItems.length === 0) {
            bucketTableBody.innerHTML = `
                <tr>
                    <td colspan="5" class="empty-state">
                        <div class="empty-state-icon">🛒</div>
                        <div class="empty-state-text">Stock bucket is empty. Select an NSE equity above and click 'Add to Bucket'.</div>
                    </td>
                </tr>`;
            updateBucketTotals();
            return;
        }

        if (bucketAllocationMode === "EQUAL_WEIGHT") {
            const eq = 1.0 / bucketItems.length;
            bucketItems.forEach(it => { it.weight = eq; });
        }

        bucketTableBody.innerHTML = bucketItems.map(item => {
            const company = STOCK_COMPANIES[item.symbol] || "NSE Equity";
            const pct = (item.weight * 100).toFixed(1);

            let weightCell = `<span class="mono-cell" style="font-weight: 600;">${pct}%</span>`;
            if (bucketAllocationMode === "CUSTOM") {
                weightCell = `
                    <div style="display: flex; align-items: center; gap: 0.35rem;">
                        <input type="number" min="1" max="100" step="1" class="control-input bucket-weight-input"
                            data-symbol="${item.symbol}" value="${Math.round(item.weight * 100)}"
                            style="width: 70px; padding: 0.25rem 0.5rem; text-align: right;">
                        <span style="color: var(--text-muted); font-size: 0.85rem;">%</span>
                    </div>`;
            }

            return `
                <tr>
                    <td><strong>${item.symbol}</strong></td>
                    <td>${company}</td>
                    <td>${weightCell}</td>
                    <td class="mono-cell">${item.weight.toFixed(4)}</td>
                    <td>
                        <button class="btn btn-outline btn-sm btn-remove-bucket" data-symbol="${item.symbol}" title="Remove from bucket">
                            ✕
                        </button>
                    </td>
                </tr>`;
        }).join("");

        // Attach weight input listeners
        document.querySelectorAll(".bucket-weight-input").forEach(inp => {
            inp.addEventListener("input", (e) => {
                const sym = e.target.getAttribute("data-symbol");
                const val = parseFloat(e.target.value) || 0;
                const it = bucketItems.find(x => x.symbol === sym);
                if (it) {
                    it.weight = val / 100.0;
                    updateBucketTotals();
                }
            });
        });

        // Attach remove button listeners
        document.querySelectorAll(".btn-remove-bucket").forEach(btn => {
            btn.addEventListener("click", () => {
                const sym = btn.getAttribute("data-symbol");
                bucketItems = bucketItems.filter(x => x.symbol !== sym);
                renderBucket();
            });
        });

        updateBucketTotals();
    }

    if (btnAddToBucket && bucketStockSelect) {
        btnAddToBucket.addEventListener("click", () => {
            const sym = bucketStockSelect.value;
            if (bucketItems.some(x => x.symbol === sym)) {
                alert(`Stock ${sym} is already in the bucket.`);
                return;
            }
            bucketItems.push({ symbol: sym, weight: 0.0 });
            renderBucket();
        });
    }

    if (btnEqualizeBucketWeights) {
        btnEqualizeBucketWeights.addEventListener("click", () => {
            bucketAllocationMode = "EQUAL_WEIGHT";
            if (radioModeEqual) radioModeEqual.checked = true;
            renderBucket();
        });
    }

    if (btnClearBucket) {
        btnClearBucket.addEventListener("click", () => {
            bucketItems = [];
            renderBucket();
        });
    }

    if (radioModeEqual) {
        radioModeEqual.addEventListener("change", () => {
            if (radioModeEqual.checked) {
                bucketAllocationMode = "EQUAL_WEIGHT";
                renderBucket();
            }
        });
    }

    if (radioModeCustom) {
        radioModeCustom.addEventListener("change", () => {
            if (radioModeCustom.checked) {
                bucketAllocationMode = "CUSTOM";
                renderBucket();
            }
        });
    }

    if (btnCreatePortfolio) {
        btnCreatePortfolio.addEventListener("click", async () => {
            if (createPortfolioMsg) createPortfolioMsg.style.display = "none";

            const name = (inputPortfolioName ? inputPortfolioName.value : "").trim();
            if (!name) {
                if (createPortfolioMsg) {
                    createPortfolioMsg.textContent = "Please enter a unique portfolio name.";
                    createPortfolioMsg.style.display = "block";
                }
                return;
            }

            if (bucketItems.length === 0) {
                if (createPortfolioMsg) {
                    createPortfolioMsg.textContent = "Cannot create portfolio with empty stock bucket.";
                    createPortfolioMsg.style.display = "block";
                }
                return;
            }

            if (bucketAllocationMode === "CUSTOM" && !updateBucketTotals()) {
                if (createPortfolioMsg) {
                    createPortfolioMsg.textContent = "Target allocations must sum to 100% in custom allocation mode.";
                    createPortfolioMsg.style.display = "block";
                }
                return;
            }

            const initialCash = parseFloat(inputPortfolioCapital ? inputPortfolioCapital.value : "100000") || 100000;
            const strategy = selectPortfolioStrategy ? selectPortfolioStrategy.value : "BUY_AND_HOLD";
            const risk = selectPortfolioRisk ? selectPortfolioRisk.value : "MODERATE";

            btnCreatePortfolio.disabled = true;
            btnCreatePortfolio.innerHTML = `<span>⏳</span> Creating...`;

            try {
                const res = await fetch("/api/v1/trading/portfolios", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        name: name,
                        initial_cash: initialCash,
                        allocation_method: bucketAllocationMode,
                        strategy_type: strategy,
                        risk_config: { profile: risk },
                        bucket: bucketItems.map(it => ({ symbol: it.symbol, target_weight: it.weight })),
                    }),
                });
                const result = await res.json();

                if (res.ok && result.success) {
                    currentActivePortfolioName = name;
                    if (inputPortfolioName) inputPortfolioName.value = "";
                    await loadPortfoliosList();
                    await loadActivePortfolio(name);
                    alert(`Portfolio "${name}" created successfully with ${bucketItems.length} bucket stocks!`);
                    const simSection = document.getElementById("simulation-card");
                    if (simSection) simSection.scrollIntoView({ behavior: "smooth" });
                } else {
                    const err = result.error ? (result.error.message || result.error) : "Creation failed.";
                    if (createPortfolioMsg) {
                        createPortfolioMsg.textContent = err;
                        createPortfolioMsg.style.display = "block";
                    }
                }
            } catch (err) {
                if (createPortfolioMsg) {
                    createPortfolioMsg.textContent = `Network error: ${err.message}`;
                    createPortfolioMsg.style.display = "block";
                }
            } finally {
                btnCreatePortfolio.disabled = false;
                btnCreatePortfolio.innerHTML = `<span>🚀</span> Create & Activate Portfolio`;
            }
        });
    }

    // -------------------------------------------------------------------------
    // Portfolio Dashboard & Switcher Integration
    // -------------------------------------------------------------------------
    async function loadPortfoliosList() {
        if (!selectActivePortfolio) return;
        try {
            const res = await fetch("/api/v1/trading/portfolios", { cache: "no-store" });
            const result = await res.json();
            if (res.ok && result.success && Array.isArray(result.data)) {
                selectActivePortfolio.innerHTML = result.data.map(p => {
                    const selected = p.name === currentActivePortfolioName ? "selected" : "";
                    const desc = p.name === "default" ? " (Default Paper Portfolio)" : "";
                    return `<option value="${p.name}" ${selected}>${p.name}${desc}</option>`;
                }).join("");

                if (btnDeleteActivePortfolio) {
                    btnDeleteActivePortfolio.disabled = (currentActivePortfolioName === "default");
                }
            }
        } catch (err) {
            console.error("Error loading portfolios list:", err);
        }
    }

    if (selectActivePortfolio) {
        selectActivePortfolio.addEventListener("change", () => {
            currentActivePortfolioName = selectActivePortfolio.value;
            if (btnDeleteActivePortfolio) {
                btnDeleteActivePortfolio.disabled = (currentActivePortfolioName === "default");
            }
            loadActivePortfolio(currentActivePortfolioName);
        });
    }

    if (btnDeleteActivePortfolio) {
        btnDeleteActivePortfolio.addEventListener("click", async () => {
            if (currentActivePortfolioName === "default") {
                alert("The default portfolio cannot be deleted.");
                return;
            }
            if (!confirm(`Are you sure you want to delete portfolio "${currentActivePortfolioName}"? All its positions and trades will be removed.`)) {
                return;
            }

            try {
                const res = await fetch(`/api/v1/trading/portfolios/${currentActivePortfolioName}`, { method: "DELETE" });
                const result = await res.json();
                if (res.ok && result.success) {
                    currentActivePortfolioName = "default";
                    await loadPortfoliosList();
                    await loadActivePortfolio("default");
                    alert("Portfolio deleted.");
                } else {
                    alert(`Failed to delete: ${result.error || "Unknown error"}`);
                }
            } catch (err) {
                alert(`Network error deleting portfolio: ${err.message}`);
            }
        });
    }

    if (btnRebalancePortfolio) {
        btnRebalancePortfolio.addEventListener("click", async () => {
            btnRebalancePortfolio.disabled = true;
            btnRebalancePortfolio.innerHTML = `<span>⏳</span> Rebalancing...`;

            try {
                const res = await fetch(`/api/v1/trading/portfolios/${currentActivePortfolioName}/rebalance`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({}),
                });
                const result = await res.json();
                if (res.ok && result.success && result.data) {
                    const d = result.data;
                    alert(`Rebalance completed! ${d.trades_count} trade orders executed across bucket stocks.`);
                    await loadActivePortfolio(currentActivePortfolioName);
                } else {
                    const err = result.error ? (result.error.message || result.error) : "Rebalance failed.";
                    alert(`Rebalance error: ${err}`);
                }
            } catch (err) {
                alert(`Network error during rebalance: ${err.message}`);
            } finally {
                btnRebalancePortfolio.disabled = false;
                btnRebalancePortfolio.innerHTML = `<span>🔄</span> Rebalance Portfolio`;
            }
        });
    }

    async function loadEquityCurve(portfolioName) {
        if (!portfolioEquityChart) return;
        try {
            const res = await fetch(`/api/v1/trading/portfolios/${portfolioName}/snapshots`, { cache: "no-store" });
            const result = await res.json();
            const snapshots = (result.ok && result.success && Array.isArray(result.data)) ? result.data : [];

            if (snapshots.length === 0) {
                portfolioEquityChart.innerHTML = `
                    <div class="empty-state" style="padding: 2.5rem 1rem;">
                        <div class="empty-state-icon">⏳</div>
                        <div class="empty-state-text">No valuation snapshots recorded yet for '${portfolioName}'. Click 'Rebalance Portfolio' or 'Execute Single-Asset Step' to start.</div>
                    </div>`;
                return;
            }

            if (typeof Plotly === "undefined") {
                portfolioEquityChart.innerHTML = `<div class="empty-state">Plotly.js loading...</div>`;
                return;
            }

            const x = snapshots.map(s => s.timestamp.replace("T", " ").substring(0, 19));
            const y = snapshots.map(s => s.portfolio_value);

            const trace = {
                x: x,
                y: y,
                type: "scatter",
                mode: "lines+markers",
                name: "Portfolio Value",
                line: { color: "#10b981", width: 2.5 },
                marker: { size: 4, color: "#10b981" },
                fill: "tozeroy",
                fillcolor: "rgba(16, 185, 129, 0.08)",
                hovertemplate: "%{x}<br>Value: ₹%{y:,.2f}<extra></extra>",
            };

            const layout = {
                paper_bgcolor: "#0a0e17",
                plot_bgcolor: "#0a0e17",
                margin: { t: 20, r: 20, b: 35, l: 60 },
                xaxis: {
                    showgrid: true,
                    gridcolor: "rgba(255, 255, 255, 0.05)",
                    tickfont: { color: "#94a3b8", size: 10 },
                },
                yaxis: {
                    showgrid: true,
                    gridcolor: "rgba(255, 255, 255, 0.05)",
                    tickfont: { color: "#94a3b8", size: 10 },
                    tickprefix: "₹",
                },
                hovermode: "x unified",
                showlegend: false,
            };

            Plotly.newPlot(portfolioEquityChart, [trace], layout, { responsive: true, displayModeBar: false });
        } catch (err) {
            console.error("Error loading equity curve:", err);
        }
    }

    async function loadActivePortfolio(portfolioName = "default") {
        try {
            const res = await fetch(`/api/v1/trading/portfolios/${portfolioName}`, { cache: "no-store" });
            const result = await res.json();
            if (res.ok && result.success && result.data) {
                const p = result.data;
                if (simCash) simCash.textContent = formatCurrencyINR(p.cash);
                if (simTotalValue) simTotalValue.textContent = formatCurrencyINR(p.portfolio_value);
                if (simPeakDd) {
                    const peakStr = formatCurrencyINR(p.peak_value);
                    const ddPct = (p.drawdown_pct !== undefined ? p.drawdown_pct : 0.0).toFixed(1);
                    simPeakDd.textContent = `${peakStr} (${ddPct}%)`;
                }
                if (simTotalPnl) {
                    const pnlVal = p.total_pnl !== undefined ? p.total_pnl : (p.portfolio_value - p.initial_cash);
                    const retPct = p.return_pct !== undefined ? p.return_pct : 0.0;
                    simTotalPnl.textContent = `${formatCurrencyINR(pnlVal)} (${retPct >= 0 ? "+" : ""}${retPct.toFixed(2)}%)`;
                    simTotalPnl.className = retPct >= 0 ? "stat-value text-emerald" : "stat-value text-rose";
                }

                // Render Bucket vs Holdings Comparison
                if (holdingsTableBody) {
                    const bucket = p.bucket || [];
                    const positions = p.positions || [];
                    const posMap = {};
                    positions.forEach(pos => { posMap[pos.symbol] = pos; });

                    if (bucket.length === 0 && positions.length === 0) {
                        holdingsTableBody.innerHTML = `
                            <tr>
                                <td colspan="7" class="empty-state">
                                    <div class="empty-state-icon">💼</div>
                                    <div class="empty-state-text">No bucket or positions configured for this portfolio.</div>
                                </td>
                            </tr>`;
                    } else {
                        // All symbols across bucket and positions
                        const allSyms = Array.from(new Set([...bucket.map(b => b.symbol), ...positions.map(pos => pos.symbol)]));
                        holdingsTableBody.innerHTML = allSyms.map(sym => {
                            const bItem = bucket.find(b => b.symbol === sym);
                            const pos = posMap[sym];
                            const targetWeightStr = bItem ? `${(bItem.target_weight * 100).toFixed(1)}%` : "0.0%";
                            const qty = pos ? pos.quantity : 0;
                            const price = pos ? formatCurrencyINR(pos.current_price) : "--";
                            const mktVal = pos ? formatCurrencyINR(pos.market_value) : "₹0.00";
                            const actualWeight = (pos && p.portfolio_value > 0)
                                ? `${((pos.market_value / p.portfolio_value) * 100).toFixed(1)}%`
                                : "0.0%";
                            const pnl = pos ? formatCurrencyINR(pos.unrealized_pnl) : "₹0.00";
                            const pnlClass = pos && pos.unrealized_pnl >= 0 ? "text-emerald" : (pos && pos.unrealized_pnl < 0 ? "text-rose" : "");

                            return `
                                <tr>
                                    <td><strong>${sym}</strong></td>
                                    <td class="mono-cell">${targetWeightStr}</td>
                                    <td class="mono-cell">${qty}</td>
                                    <td class="mono-cell">${price}</td>
                                    <td class="mono-cell">${mktVal}</td>
                                    <td class="mono-cell">${actualWeight}</td>
                                    <td class="mono-cell ${pnlClass}">${pnl}</td>
                                </tr>`;
                        }).join("");
                    }
                }

                // Update single-step symbol selector
                if (selectSimSymbol && p.bucket && p.bucket.length > 0) {
                    const currentVal = selectSimSymbol.value;
                    const hasCurrent = p.bucket.some(b => b.symbol === currentVal);
                    selectSimSymbol.innerHTML = p.bucket.map(b => {
                        const sel = (b.symbol === currentVal || (!hasCurrent && b.symbol === p.bucket[0].symbol)) ? "selected" : "";
                        return `<option value="${b.symbol}" ${sel}>${b.symbol}</option>`;
                    }).join("");
                }

                // Load associated trades, risk events, and equity curve
                await Promise.all([
                    loadTrades(portfolioName),
                    loadRiskEvents(portfolioName),
                    loadEquityCurve(portfolioName),
                ]);
            }
        } catch (err) {
            console.error("Error loading active portfolio:", err);
        }
    }

    async function loadTrades(portfolioName = "default") {
        if (!tradesTableBody) return;
        try {
            const res = await fetch(`/api/v1/trading/trades?portfolio=${portfolioName}&limit=15`, { cache: "no-store" });
            const result = await res.json();
            const trades = (result.ok && result.success && Array.isArray(result.data))
                ? result.data
                : (result.data && result.data.trades ? result.data.trades : []);

            if (trades.length === 0) {
                tradesTableBody.innerHTML = `
                    <tr>
                        <td colspan="8" class="empty-state">
                            <div class="empty-state-icon">📝</div>
                            <div class="empty-state-text">No simulated trades recorded for '${portfolioName}'. Click 'Rebalance' or 'Execute Single-Asset Step'.</div>
                        </td>
                    </tr>`;
                return;
            }

            tradesTableBody.innerHTML = trades.map(t => {
                const actionBadge = t.action === "BUY"
                    ? `<span class="badge badge-success">BUY</span>`
                    : (t.action === "SELL" ? `<span class="badge badge-danger">SELL</span>` : `<span class="badge badge-neutral">${t.action}</span>`);
                const grossVal = t.price * t.quantity;

                return `
                    <tr>
                        <td class="mono-cell">${t.timestamp.replace("T", " ").substring(0, 19)}</td>
                        <td><strong>${t.symbol}</strong></td>
                        <td>${actionBadge}</td>
                        <td class="mono-cell">${t.quantity}</td>
                        <td class="mono-cell">${formatCurrencyINR(t.price)}</td>
                        <td class="mono-cell">${formatCurrencyINR(grossVal)}</td>
                        <td class="mono-cell">${formatCurrencyINR(t.fee)}</td>
                        <td><span class="badge badge-success">${t.risk_status}</span></td>
                    </tr>`;
            }).join("");
        } catch (err) {
            console.error("Error loading trades:", err);
        }
    }

    async function loadRiskEvents(portfolioName = "default") {
        if (!riskEventsTableBody) return;
        try {
            const res = await fetch(`/api/v1/trading/risk-events?portfolio=${portfolioName}&limit=15`, { cache: "no-store" });
            const result = await res.json();
            const events = (result.ok && result.success && Array.isArray(result.data))
                ? result.data
                : (result.data && result.data.risk_events ? result.data.risk_events : []);

            if (events.length === 0) {
                riskEventsTableBody.innerHTML = `
                    <tr>
                        <td colspan="7" class="empty-state">
                            <div class="empty-state-icon">🛡️</div>
                            <div class="empty-state-text">No risk intervention events. The risk manager operates as a protective gateway.</div>
                        </td>
                    </tr>`;
                return;
            }

            riskEventsTableBody.innerHTML = events.map(e => {
                const sevBadge = e.severity === "BLOCKED"
                    ? `<span class="badge badge-danger">BLOCKED</span>`
                    : `<span class="badge badge-warning">${e.severity}</span>`;

                return `
                    <tr>
                        <td class="mono-cell">${e.timestamp.replace("T", " ").substring(0, 19)}</td>
                        <td><strong>${e.symbol}</strong></td>
                        <td><span class="mono-cell" style="color:#fbbf24;">${e.rule_breached}</span></td>
                        <td>${sevBadge}</td>
                        <td class="mono-cell">${e.proposed_action}</td>
                        <td class="mono-cell">${e.effective_action}</td>
                        <td style="font-size:0.8rem; color:var(--text-muted);">${e.details || "--"}</td>
                    </tr>`;
            }).join("");
        } catch (err) {
            console.error("Error loading risk events:", err);
        }
    }

    if (btnRunSimStep) {
        btnRunSimStep.addEventListener("click", async () => {
            const symbol = selectSimSymbol ? selectSimSymbol.value : "RELIANCE.NS";
            btnRunSimStep.disabled = true;
            if (simSpinner) simSpinner.style.display = "inline";

            try {
                const res = await fetch("/api/v1/trading/step", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ symbol: symbol, portfolio: currentActivePortfolioName }),
                });
                const result = await res.json();

                if (res.ok && result.success && result.data) {
                    const d = result.data;
                    if (stepOutcomeCard) stepOutcomeCard.style.display = "block";

                    if (outcomePpoAction) outcomePpoAction.textContent = d.decision.proposed_action;
                    if (outcomeRiskStatus) {
                        outcomeRiskStatus.textContent = d.decision.risk_status;
                        outcomeRiskStatus.className = d.decision.risk_status === "APPROVED"
                            ? "stat-value text-emerald"
                            : (d.decision.risk_status === "BLOCKED" ? "stat-value text-rose" : "stat-value text-amber");
                    }
                    if (outcomeEffectiveAction) {
                        outcomeEffectiveAction.textContent = d.decision.effective_action;
                    }
                    if (outcomeRegime) {
                        outcomeRegime.textContent = d.regime ? d.regime.regime_label : "Standard";
                    }
                    if (outcomeDetails) {
                        outcomeDetails.innerHTML = `Bar: ${d.timestamp} | Benchmark Close: ${formatCurrencyINR(d.current_price)} | Model ID: ${d.model ? d.model.model_id : "--"}`;
                    }

                    // Reload updated state for active portfolio
                    await loadActivePortfolio(currentActivePortfolioName);
                } else {
                    const msg = result.error ? (result.error.message || result.error) : "Simulation step error";
                    alert(`Simulation step error: ${msg}`);
                }
            } catch (err) {
                alert(`Network error executing step: ${err.message}`);
            } finally {
                btnRunSimStep.disabled = false;
                if (simSpinner) simSpinner.style.display = "none";
            }
        });
    }

    if (btnCheckHealth) {
        btnCheckHealth.addEventListener("click", () => {
            fetchHealthStatus();
        });
    }

    // -------------------------------------------------------------------------
    // 5. Interactive Candlestick & Technical Indicator Visualization (Phase 10A)
    // -------------------------------------------------------------------------
    const chartSymbolSelect = document.getElementById("chart-symbol-select");
    const periodPresets = document.getElementById("period-presets");
    const chartStartDate = document.getElementById("chart-start-date");
    const chartEndDate = document.getElementById("chart-end-date");
    const btnApplyDates = document.getElementById("btn-apply-dates");
    const toggleSMA5 = document.getElementById("toggle-sma5");
    const toggleSMA20 = document.getElementById("toggle-sma20");
    const toggleRSI = document.getElementById("toggle-rsi");
    const toggleMACD = document.getElementById("toggle-macd");
    const btnChartResetZoom = document.getElementById("btn-chart-reset-zoom");
    const chartMetaSymbol = document.getElementById("chart-meta-symbol");
    const chartMetaPrice = document.getElementById("chart-meta-price");
    const chartMetaChange = document.getElementById("chart-meta-change");
    const chartMetaRange = document.getElementById("chart-meta-range");
    const chartMetaBars = document.getElementById("chart-meta-bars");
    const chartErrorBanner = document.getElementById("chart-error-banner");
    const chartLoadingOverlay = document.getElementById("chart-loading-overlay");
    const chartLoadingText = document.getElementById("chart-loading-text");
    const chartEmptyState = document.getElementById("chart-empty-state");
    const marketChartElem = document.getElementById("market-candlestick-chart");

    let currentChartSymbol = "RELIANCE.NS";
    let currentPeriodPreset = "1Y";
    let chartCachedBars = [];
    let chartAbortController = null;

    function getPresetDateRange(period) {
        const maxDate = new Date("2025-12-30T23:59:59Z");
        let startDt = new Date(maxDate);

        if (period === "1M") {
            startDt.setDate(startDt.getDate() - 30);
        } else if (period === "3M") {
            startDt.setDate(startDt.getDate() - 90);
        } else if (period === "6M") {
            startDt.setDate(startDt.getDate() - 180);
        } else if (period === "1Y") {
            startDt.setDate(startDt.getDate() - 365);
        } else if (period === "5Y") {
            startDt.setDate(startDt.getDate() - (365 * 5));
        } else if (period === "ALL") {
            return { startDate: "2016-01-01", endDate: "2025-12-31" };
        }

        const formatIsoDate = (d) => d.toISOString().split("T")[0];
        return {
            startDate: formatIsoDate(startDt),
            endDate: formatIsoDate(maxDate),
        };
    }

    async function fetchAndRenderChart() {
        if (!marketChartElem) return;

        if (chartAbortController) {
            chartAbortController.abort();
        }
        chartAbortController = new AbortController();

        if (chartLoadingOverlay) {
            chartLoadingOverlay.style.display = "flex";
            if (chartLoadingText) {
                chartLoadingText.textContent = `Loading ${currentChartSymbol} candlestick series...`;
            }
        }
        if (chartErrorBanner) chartErrorBanner.style.display = "none";
        if (chartEmptyState) chartEmptyState.style.display = "none";

        let startParam = chartStartDate ? chartStartDate.value.trim() : "";
        let endParam = chartEndDate ? chartEndDate.value.trim() : "";

        const queryParams = new URLSearchParams({
            limit: "5000",
            indicators: "true",
        });
        if (startParam) queryParams.set("start_date", startParam);
        if (endParam) queryParams.set("end_date", endParam);

        try {
            const res = await fetch(`/api/v1/assets/${currentChartSymbol}/history?${queryParams.toString()}`, {
                signal: chartAbortController.signal,
                cache: "no-store",
            });
            const result = await res.json();

            if (!res.ok || !result.success) {
                throw new Error(result.error || result.message || `Failed to fetch data (${res.status})`);
            }

            const bars = result.data || [];
            chartCachedBars = bars;

            if (bars.length === 0) {
                if (chartEmptyState) chartEmptyState.style.display = "block";
                if (chartMetaBars) chartMetaBars.textContent = "0 bars";
                if (chartMetaRange) chartMetaRange.textContent = "No data in range";
                if (typeof Plotly !== "undefined") {
                    Plotly.purge(marketChartElem);
                }
                return;
            }

            updateChartTickerMeta(bars);
            renderPlotlyTraces(bars);

        } catch (err) {
            if (err.name === "AbortError") {
                return;
            }
            console.error("Error loading chart data:", err);
            if (chartErrorBanner) {
                chartErrorBanner.style.display = "flex";
                chartErrorBanner.innerHTML = `<span>⚠️</span><div><strong>Chart Load Error:</strong> ${err.message}</div>`;
            }
            if (chartEmptyState) chartEmptyState.style.display = "block";
            if (typeof Plotly !== "undefined") {
                Plotly.purge(marketChartElem);
            }
        } finally {
            if (chartLoadingOverlay) {
                chartLoadingOverlay.style.display = "none";
            }
        }
    }

    function updateChartTickerMeta(bars) {
        if (!bars || bars.length === 0) return;
        const latest = bars[bars.length - 1];
        const first = bars[0];
        const prev = bars.length > 1 ? bars[bars.length - 2] : latest;

        if (chartMetaSymbol) chartMetaSymbol.textContent = currentChartSymbol;

        const latestClose = Number(latest.close);
        const prevClose = Number(prev.close);
        const change = latestClose - prevClose;
        const changePct = prevClose > 0 ? (change / prevClose) * 100 : 0;

        if (chartMetaPrice) chartMetaPrice.textContent = formatCurrencyINR(latestClose);

        if (chartMetaChange) {
            const sign = change >= 0 ? "+" : "";
            const changeStr = `${sign}${formatCurrencyINR(change)} (${sign}${changePct.toFixed(2)}%)`;
            chartMetaChange.textContent = changeStr;
            chartMetaChange.className = change >= 0 ? "meta-ticker-change text-emerald" : "meta-ticker-change text-rose";
        }

        if (chartMetaRange) {
            const startD = first.timestamp.split("T")[0];
            const endD = latest.timestamp.split("T")[0];
            chartMetaRange.textContent = `${startD} → ${endD}`;
        }

        if (chartMetaBars) {
            chartMetaBars.textContent = `${bars.length.toLocaleString()} bars`;
        }
    }

    function renderPlotlyTraces(bars) {
        if (!marketChartElem || typeof Plotly === "undefined") return;

        const showSMA5 = toggleSMA5 ? toggleSMA5.checked : true;
        const showSMA20 = toggleSMA20 ? toggleSMA20.checked : true;
        const showRSI = toggleRSI ? toggleRSI.checked : false;
        const showMACD = toggleMACD ? toggleMACD.checked : false;

        const dates = bars.map(b => b.timestamp);
        const opens = bars.map(b => Number(b.open));
        const highs = bars.map(b => Number(b.high));
        const lows = bars.map(b => Number(b.low));
        const closes = bars.map(b => Number(b.close));
        const volumes = bars.map(b => Number(b.volume));

        const hoverTexts = bars.map(b => {
            const dateStr = b.timestamp.split("T")[0];
            return `<b>${dateStr}</b><br>` +
                   `Open: ₹${Number(b.open).toFixed(2)}<br>` +
                   `High: ₹${Number(b.high).toFixed(2)}<br>` +
                   `Low: ₹${Number(b.low).toFixed(2)}<br>` +
                   `Close: ₹${Number(b.close).toFixed(2)}<br>` +
                   `Volume: ${Number(b.volume).toLocaleString()}`;
        });

        let priceDomain, volumeDomain, rsiDomain, macdDomain;
        let totalPanels = 2;
        if (showRSI) totalPanels += 1;
        if (showMACD) totalPanels += 1;

        if (totalPanels === 2) {
            priceDomain = [0.28, 1.0];
            volumeDomain = [0.0, 0.22];
        } else if (totalPanels === 3) {
            priceDomain = [0.46, 1.0];
            volumeDomain = [0.26, 0.40];
            if (showRSI) rsiDomain = [0.0, 0.20];
            if (showMACD) macdDomain = [0.0, 0.20];
        } else if (totalPanels === 4) {
            priceDomain = [0.55, 1.0];
            volumeDomain = [0.38, 0.49];
            rsiDomain = [0.20, 0.32];
            macdDomain = [0.0, 0.14];
        }

        const traces = [];

        // 1. Candlestick Trace
        traces.push({
            type: "candlestick",
            x: dates,
            open: opens,
            high: highs,
            low: lows,
            close: closes,
            name: `${currentChartSymbol}`,
            text: hoverTexts,
            hoverinfo: "text",
            increasing: { line: { color: "#10b981", width: 1.5 }, fillcolor: "#10b981" },
            decreasing: { line: { color: "#f43f5e", width: 1.5 }, fillcolor: "#f43f5e" },
            yaxis: "y",
            xaxis: "x",
        });

        // 2. SMA 5 Trace
        if (showSMA5) {
            const sma5Vals = bars.map(b => b.sma_5 !== null && b.sma_5 !== undefined ? Number(b.sma_5) : null);
            traces.push({
                type: "scatter",
                mode: "lines",
                x: dates,
                y: sma5Vals,
                name: "SMA 5",
                line: { color: "#818cf8", width: 1.6 },
                yaxis: "y",
                xaxis: "x",
                hoverinfo: "y+name",
            });
        }

        // 3. SMA 20 Trace
        if (showSMA20) {
            const sma20Vals = bars.map(b => b.sma_20 !== null && b.sma_20 !== undefined ? Number(b.sma_20) : null);
            traces.push({
                type: "scatter",
                mode: "lines",
                x: dates,
                y: sma20Vals,
                name: "SMA 20",
                line: { color: "#fbbf24", width: 1.6 },
                yaxis: "y",
                xaxis: "x",
                hoverinfo: "y+name",
            });
        }

        // 4. Volume Bar Trace
        const volColors = bars.map((b, i) => {
            const isUp = closes[i] >= opens[i];
            return isUp ? "rgba(16, 185, 129, 0.45)" : "rgba(244, 63, 94, 0.45)";
        });

        traces.push({
            type: "bar",
            x: dates,
            y: volumes,
            name: "Volume",
            marker: { color: volColors },
            yaxis: "y2",
            xaxis: "x",
            hoverinfo: "x+y+name",
        });

        // 5. RSI 14 Trace
        let rsiYAxisRef = "y3";
        if (showRSI) {
            const rsiVals = bars.map(b => b.rsi_14 !== null && b.rsi_14 !== undefined ? Number(b.rsi_14) : null);
            traces.push({
                type: "scatter",
                mode: "lines",
                x: dates,
                y: rsiVals,
                name: "RSI 14",
                line: { color: "#06b6d4", width: 1.5 },
                yaxis: rsiYAxisRef,
                xaxis: "x",
                hoverinfo: "y+name",
            });
        }

        // 6. MACD Trace
        let macdYAxisRef = showRSI ? "y4" : "y3";
        if (showMACD) {
            const macdLines = bars.map(b => b.macd_line !== null && b.macd_line !== undefined ? Number(b.macd_line) : null);
            const macdSignals = bars.map(b => b.macd_signal !== null && b.macd_signal !== undefined ? Number(b.macd_signal) : null);
            const macdHists = bars.map(b => b.macd_histogram !== null && b.macd_histogram !== undefined ? Number(b.macd_histogram) : null);

            const histColors = macdHists.map(h => (h !== null && h >= 0) ? "rgba(16, 185, 129, 0.6)" : "rgba(244, 63, 94, 0.6)");

            traces.push({
                type: "bar",
                x: dates,
                y: macdHists,
                name: "MACD Hist",
                marker: { color: histColors },
                yaxis: macdYAxisRef,
                xaxis: "x",
                hoverinfo: "y+name",
            });

            traces.push({
                type: "scatter",
                mode: "lines",
                x: dates,
                y: macdLines,
                name: "MACD Line",
                line: { color: "#38bdf8", width: 1.5 },
                yaxis: macdYAxisRef,
                xaxis: "x",
                hoverinfo: "y+name",
            });

            traces.push({
                type: "scatter",
                mode: "lines",
                x: dates,
                y: macdSignals,
                name: "Signal",
                line: { color: "#fb923c", width: 1.5 },
                yaxis: macdYAxisRef,
                xaxis: "x",
                hoverinfo: "y+name",
            });
        }

        const shapes = [];
        if (showRSI) {
            shapes.push({
                type: "line",
                xref: "paper",
                x0: 0,
                x1: 1,
                yref: rsiYAxisRef,
                y0: 70,
                y1: 70,
                line: { color: "rgba(244, 63, 94, 0.4)", width: 1, dash: "dot" },
            });
            shapes.push({
                type: "line",
                xref: "paper",
                x0: 0,
                x1: 1,
                yref: rsiYAxisRef,
                y0: 30,
                y1: 30,
                line: { color: "rgba(16, 185, 129, 0.4)", width: 1, dash: "dot" },
            });
            shapes.push({
                type: "line",
                xref: "paper",
                x0: 0,
                x1: 1,
                yref: rsiYAxisRef,
                y0: 50,
                y1: 50,
                line: { color: "rgba(255, 255, 255, 0.15)", width: 1, dash: "dash" },
            });
        }

        const layout = {
            dragmode: "zoom",
            showlegend: true,
            legend: {
                orientation: "h",
                x: 0,
                y: 1.05,
                font: { color: "#94a3b8", size: 10, family: "Inter, sans-serif" },
                bgcolor: "rgba(15, 23, 42, 0.6)",
            },
            margin: { t: 35, r: 50, b: 35, l: 55 },
            plot_bgcolor: "#070a12",
            paper_bgcolor: "#070a12",
            font: { color: "#cbd5e1", family: "Inter, sans-serif" },
            xaxis: {
                rangeslider: { visible: false },
                type: "date",
                gridcolor: "rgba(255, 255, 255, 0.05)",
                tickfont: { size: 10, color: "#64748b" },
                spikemode: "across",
                spikesnap: "cursor",
                spikethickness: 1,
                spikecolor: "rgba(255, 255, 255, 0.2)",
            },
            yaxis: {
                title: "Price (₹)",
                titlefont: { size: 11, color: "#94a3b8" },
                tickfont: { size: 10, color: "#64748b" },
                gridcolor: "rgba(255, 255, 255, 0.05)",
                side: "right",
                domain: priceDomain,
                spikemode: "across",
                spikesnap: "cursor",
                spikethickness: 1,
                spikecolor: "rgba(255, 255, 255, 0.2)",
            },
            yaxis2: {
                title: "Vol",
                titlefont: { size: 10, color: "#94a3b8" },
                tickfont: { size: 9, color: "#64748b" },
                gridcolor: "rgba(255, 255, 255, 0.03)",
                side: "right",
                domain: volumeDomain,
            },
            shapes: shapes,
        };

        if (showRSI) {
            layout[rsiYAxisRef] = {
                title: "RSI",
                titlefont: { size: 10, color: "#06b6d4" },
                tickfont: { size: 9, color: "#64748b" },
                gridcolor: "rgba(255, 255, 255, 0.03)",
                side: "right",
                domain: rsiDomain,
                range: [0, 100],
                tickvals: [30, 50, 70],
            };
        }

        if (showMACD) {
            layout[macdYAxisRef] = {
                title: "MACD",
                titlefont: { size: 10, color: "#38bdf8" },
                tickfont: { size: 9, color: "#64748b" },
                gridcolor: "rgba(255, 255, 255, 0.03)",
                side: "right",
                domain: macdDomain,
            };
        }

        const config = {
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ["lasso2d", "select2d"],
        };

        Plotly.react(marketChartElem, traces, layout, config);
    }

    function initMarketChart() {
        if (!marketChartElem) return;

        const initialDates = getPresetDateRange(currentPeriodPreset);
        if (chartStartDate) chartStartDate.value = initialDates.startDate;
        if (chartEndDate) chartEndDate.value = initialDates.endDate;

        if (chartSymbolSelect) {
            chartSymbolSelect.addEventListener("change", (e) => {
                currentChartSymbol = e.target.value;
                fetchAndRenderChart();
            });
        }

        if (periodPresets) {
            periodPresets.querySelectorAll(".period-btn").forEach(btn => {
                btn.addEventListener("click", () => {
                    periodPresets.querySelectorAll(".period-btn").forEach(b => b.classList.remove("active"));
                    btn.classList.add("active");

                    currentPeriodPreset = btn.getAttribute("data-period");
                    const range = getPresetDateRange(currentPeriodPreset);
                    if (chartStartDate) chartStartDate.value = range.startDate;
                    if (chartEndDate) chartEndDate.value = range.endDate;

                    fetchAndRenderChart();
                });
            });
        }

        if (btnApplyDates) {
            btnApplyDates.addEventListener("click", () => {
                if (periodPresets) {
                    periodPresets.querySelectorAll(".period-btn").forEach(b => b.classList.remove("active"));
                }
                currentPeriodPreset = "CUSTOM";
                fetchAndRenderChart();
            });
        }

        const indicatorToggles = [toggleSMA5, toggleSMA20, toggleRSI, toggleMACD];
        indicatorToggles.forEach(t => {
            if (t) {
                t.addEventListener("change", () => {
                    const parentLabel = t.closest(".indicator-label");
                    if (parentLabel) {
                        parentLabel.classList.toggle("active", t.checked);
                    }
                    if (chartCachedBars && chartCachedBars.length > 0) {
                        renderPlotlyTraces(chartCachedBars);
                    } else {
                        fetchAndRenderChart();
                    }
                });
            }
        });

        if (btnChartResetZoom) {
            btnChartResetZoom.addEventListener("click", () => {
                if (marketChartElem && typeof Plotly !== "undefined") {
                    Plotly.relayout(marketChartElem, {
                        "xaxis.autorange": true,
                        "yaxis.autorange": true,
                        "yaxis2.autorange": true,
                        "yaxis3.autorange": true,
                        "yaxis4.autorange": true,
                    });
                }
            });
        }

        window.addEventListener("resize", () => {
            if (marketChartElem && typeof Plotly !== "undefined" && marketChartElem.data) {
                Plotly.Plots.resize(marketChartElem);
            }
        });

        fetchAndRenderChart();
    }

    // =========================================================================
    // Phase 11F: Live Market Data & Intraday Candlestick Dashboard Controller
    // =========================================================================
    function initLiveMarketDashboard() {
        const btnLiveConnect = document.getElementById("btn-live-connect");
        const btnLiveDisconnect = document.getElementById("btn-live-disconnect");
        const liveConnBadge = document.getElementById("live-conn-badge");
        const liveTickerStrip = document.getElementById("live-ticker-strip");
        const liveSymbolSelect = document.getElementById("live-symbol-select");
        const liveIntervalGroup = document.getElementById("live-interval-group");
        const liveChartElem = document.getElementById("live-candlestick-chart");
        const btnLiveTogglePoll = document.getElementById("btn-live-toggle-poll");
        const pollIndicator = document.getElementById("poll-indicator");

        const liveStatFreshness = document.getElementById("live-stat-freshness");
        const liveStatLatency = document.getElementById("live-stat-latency");
        const liveStatEventTime = document.getElementById("live-stat-event-time");
        const liveStatFeatures = document.getElementById("live-stat-features");
        const liveFeatureAlert = document.getElementById("live-feature-alert");

        let liveSymbol = "RELIANCE.NS";
        let liveInterval = "5m";
        let isPolling = true;
        let pollTimer = null;

        async function fetchLiveDashboard() {
            try {
                const res = await fetch(`/api/v1/realtime/dashboard?symbol=${encodeURIComponent(liveSymbol)}&interval=${encodeURIComponent(liveInterval)}`);
                const json = await res.json();
                if (!json.success || !json.data) return;

                const data = json.data;

                // 1. Connection status badge & buttons
                if (liveConnBadge) {
                    const st = data.connection.status;
                    liveConnBadge.textContent = st + (data.connection.is_stale ? " (STALE)" : "");
                    if (st === "CONNECTED" && !data.connection.is_stale) {
                        liveConnBadge.style.background = "rgba(16, 185, 129, 0.2)";
                        liveConnBadge.style.color = "var(--accent-emerald)";
                        if (btnLiveConnect) btnLiveConnect.style.display = "none";
                        if (btnLiveDisconnect) btnLiveDisconnect.style.display = "inline-flex";
                    } else if (st === "STALE" || data.connection.is_stale) {
                        liveConnBadge.style.background = "rgba(244, 63, 94, 0.2)";
                        liveConnBadge.style.color = "var(--accent-rose)";
                    } else {
                        liveConnBadge.style.background = "rgba(148, 163, 184, 0.2)";
                        liveConnBadge.style.color = "var(--text-muted)";
                        if (btnLiveConnect) btnLiveConnect.style.display = "inline-flex";
                        if (btnLiveDisconnect) btnLiveDisconnect.style.display = "none";
                    }
                }

                // 2. Render Ticker Strip across subscribed stocks
                if (liveTickerStrip) {
                    const quotes = data.quotes || {};
                    const freshness = data.freshness || {};
                    const symbols = data.subscribed_symbols && data.subscribed_symbols.length > 0 
                        ? data.subscribed_symbols 
                        : Object.keys(quotes);

                    if (symbols.length === 0) {
                        liveTickerStrip.innerHTML = `<div style="font-size: 0.85rem; color: var(--text-muted); padding: 0.5rem;">No active subscriptions. Click 'Connect Feed' or subscribe via stock bucket.</div>`;
                    } else {
                        liveTickerStrip.innerHTML = symbols.map(sym => {
                            const q = quotes[sym];
                            const f = freshness[sym] || {};
                            const priceStr = q ? formatCurrencyINR(q.last_price) : "--";
                            const fStatus = f.status || "OFFLINE";
                            const badgeColor = fStatus === "FRESH" ? "var(--accent-emerald)" : (fStatus === "AGING" ? "var(--accent-amber)" : "var(--text-muted)");
                            const isSelected = sym === liveSymbol ? "border: 1px solid var(--accent-indigo);" : "border: 1px solid rgba(255,255,255,0.08);";

                            return `
                                <div class="ticker-chip" style="background: rgba(15, 23, 42, 0.7); ${isSelected} border-radius: 8px; padding: 0.5rem 0.85rem; min-width: 150px; cursor: pointer;" onclick="document.getElementById('live-symbol-select').value='${sym}'; document.getElementById('live-symbol-select').dispatchEvent(new Event('change'));">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem;">
                                        <span style="font-weight: 600; font-size: 0.85rem; color: var(--text-primary);">${sym}</span>
                                        <span style="font-size: 0.65rem; color: ${badgeColor}; font-weight: 600; text-transform: uppercase;">${fStatus}</span>
                                    </div>
                                    <div style="font-size: 1.05rem; font-weight: 700; color: var(--text-primary); font-family: var(--font-mono);">${priceStr}</div>
                                </div>
                            `;
                        }).join("");
                    }
                }

                // 3. Render Plotly Candlestick Chart for active symbol
                const candles = data.candles || [];
                if (liveChartElem && typeof Plotly !== "undefined") {
                    if (candles.length === 0) {
                        liveChartElem.innerHTML = `<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: var(--text-muted); font-size: 0.9rem;">Awaiting candle completion for ${liveSymbol} (${liveInterval})...</div>`;
                    } else {
                        const dates = candles.map(c => c.start_time);
                        const opens = candles.map(c => c.open);
                        const highs = candles.map(c => c.high);
                        const lows = candles.map(c => c.low);
                        const closes = candles.map(c => c.close);

                        const candleTrace = {
                            x: dates,
                            open: opens,
                            high: highs,
                            low: lows,
                            close: closes,
                            type: "candlestick",
                            name: `${liveSymbol} (${liveInterval})`,
                            increasing: { line: { color: "#10b981", width: 1.5 }, fillcolor: "#10b981" },
                            decreasing: { line: { color: "#f43f5e", width: 1.5 }, fillcolor: "#f43f5e" },
                        };

                        const layout = {
                            dragmode: "zoom",
                            margin: { r: 40, t: 30, b: 40, l: 60 },
                            showlegend: false,
                            xaxis: {
                                autorange: true,
                                rangeslider: { visible: false },
                                gridcolor: "rgba(255, 255, 255, 0.05)",
                                type: "date",
                            },
                            yaxis: {
                                autorange: true,
                                gridcolor: "rgba(255, 255, 255, 0.05)",
                                tickprefix: "₹",
                            },
                            plot_bgcolor: "transparent",
                            paper_bgcolor: "transparent",
                            font: { color: "#94a3b8", family: "Inter, sans-serif", size: 11 },
                        };

                        const config = { responsive: true, displayModeBar: false };
                        Plotly.react(liveChartElem, [candleTrace], layout, config);
                    }
                }

                // 4. Update Freshness & Telemetry Stats
                const fInfo = data.freshness[liveSymbol] || {};
                if (liveStatFreshness) {
                    liveStatFreshness.textContent = fInfo.status || "--";
                    liveStatFreshness.className = fInfo.status === "FRESH" ? "stat-value text-emerald" : "stat-value text-amber";
                }
                if (liveStatLatency) {
                    liveStatLatency.textContent = fInfo.latency_ms !== null && fInfo.latency_ms !== undefined ? `${fInfo.latency_ms} ms` : "--";
                }
                if (liveStatEventTime) {
                    liveStatEventTime.textContent = fInfo.event_timestamp ? fInfo.event_timestamp.replace("T", " ").substring(0, 19) + " UTC" : "--";
                }

                // 5. Update PPO Feature Readiness Alert
                const feats = data.features_readiness || {};
                if (liveStatFeatures) {
                    if (feats.ready) {
                        liveStatFeatures.textContent = "Ready (12-d Vector)";
                        liveStatFeatures.className = "stat-value text-emerald";
                    } else if (feats.timeframe_mismatch) {
                        liveStatFeatures.textContent = `Timeframe Mismatch`;
                        liveStatFeatures.className = "stat-value text-amber";
                    } else {
                        liveStatFeatures.textContent = "Warmup Incomplete";
                        liveStatFeatures.className = "stat-value text-rose";
                    }
                }
                if (liveFeatureAlert) {
                    liveFeatureAlert.textContent = feats.reason || "Evaluating PPO feature pipeline compatibility...";
                }

            } catch (err) {
                console.error("Error fetching live market dashboard payload:", err);
            }
        }

        // Attach event listeners
        if (liveSymbolSelect) {
            liveSymbolSelect.addEventListener("change", () => {
                liveSymbol = liveSymbolSelect.value;
                fetchLiveDashboard();
            });
        }

        if (liveIntervalGroup) {
            liveIntervalGroup.querySelectorAll(".period-btn").forEach(btn => {
                btn.addEventListener("click", () => {
                    liveIntervalGroup.querySelectorAll(".period-btn").forEach(b => b.classList.remove("active"));
                    btn.classList.add("active");
                    liveInterval = btn.dataset.interval;
                    fetchLiveDashboard();
                });
            });
        }

        if (btnLiveConnect) {
            btnLiveConnect.addEventListener("click", async () => {
                try {
                    btnLiveConnect.disabled = true;
                    await fetch("/api/v1/realtime/connect", { method: "POST" });
                    await fetchLiveDashboard();
                } catch (e) {
                    console.error("Connect failed:", e);
                } finally {
                    btnLiveConnect.disabled = false;
                }
            });
        }

        if (btnLiveDisconnect) {
            btnLiveDisconnect.addEventListener("click", async () => {
                try {
                    btnLiveDisconnect.disabled = true;
                    await fetch("/api/v1/realtime/disconnect", { method: "POST" });
                    await fetchLiveDashboard();
                } catch (e) {
                    console.error("Disconnect failed:", e);
                } finally {
                    btnLiveDisconnect.disabled = false;
                }
            });
        }

        if (btnLiveTogglePoll) {
            btnLiveTogglePoll.addEventListener("click", () => {
                isPolling = !isPolling;
                if (isPolling) {
                    pollIndicator.textContent = "🟢";
                    btnLiveTogglePoll.innerHTML = `<span>🟢</span> Polling Active (2s)`;
                    startPolling();
                } else {
                    pollIndicator.textContent = "⏸️";
                    btnLiveTogglePoll.innerHTML = `<span>⏸️</span> Polling Paused`;
                    stopPolling();
                }
            });
        }

        function startPolling() {
            if (pollTimer) clearInterval(pollTimer);
            pollTimer = setInterval(fetchLiveDashboard, 2000);
        }

        function stopPolling() {
            if (pollTimer) {
                clearInterval(pollTimer);
                pollTimer = null;
            }
        }

        // Initial fetch and start loop
        fetchLiveDashboard();
        startPolling();
    }

    // -------------------------------------------------------------------------
    // Initial Load Probe
    // -------------------------------------------------------------------------
    fetchHealthStatus();
    loadDatasets();
    loadModels();
    renderBucket();
    loadPortfoliosList();
    loadActivePortfolio("default");
    initMarketChart();
    initLiveMarketDashboard();
});

