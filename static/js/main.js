/**
 * TradeMind Phase 1 Client-Side Diagnostics Script
 * Interacts with /api/health and updates status metrics in real time.
 */

document.addEventListener("DOMContentLoaded", () => {
    const btnCheckHealth = document.getElementById("btn-check-health");
    const statApiStatus = document.getElementById("stat-api-status");
    const statLatency = document.getElementById("stat-latency");
    const jsonOutput = document.getElementById("health-json-output");
    const payloadTime = document.getElementById("payload-time");
    const headerStatusText = document.getElementById("header-status-text");
    const headerPulseDot = document.getElementById("header-pulse-dot");

    /**
     * Queries the TradeMind health-check endpoint and renders results.
     */
    async function fetchHealthStatus() {
        const startTime = performance.now();
        if (statApiStatus) statApiStatus.textContent = "Probing...";

        try {
            const response = await fetch("/api/health", {
                headers: { "Accept": "application/json" },
                cache: "no-store"
            });

            const latency = Math.round(performance.now() - startTime);
            const data = await response.json();

            if (response.ok && data.status === "healthy") {
                if (statApiStatus) {
                    statApiStatus.textContent = "Healthy (200 OK)";
                    statApiStatus.className = "stat-value text-emerald";
                }
                if (statLatency) statLatency.textContent = `${latency} ms`;
                if (jsonOutput) {
                    jsonOutput.innerHTML = `<code>${JSON.stringify(data, null, 2)}</code>`;
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
                throw new Error(`Unexpected status: ${response.status}`);
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

    if (btnCheckHealth) {
        btnCheckHealth.addEventListener("click", () => {
            fetchHealthStatus();
        });
    }

    // Auto-probe on initial page load
    fetchHealthStatus();
});
