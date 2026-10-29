"""Unit and Integration Tests for User-Configured Multi-Stock Portfolios & Stock Bucket/Cart.

Covers:
    1. Stock Bucket validation (supported symbols, weight normalization, positive weights, 100% sum).
    2. Rejection of invalid inputs (empty bucket, duplicate symbols, unsupported symbols, sum != 100%).
    3. Portfolio Creation with Equal Weight and Custom allocation modes.
    4. Portfolio Isolation: Cash, positions, trades, and snapshots remain strictly independent.
    5. Bucket updates (add/remove stocks, modify weights).
    6. Multi-stock simulated rebalancing with slippage, fees, and position updates.
    7. Persistence across simulated restarts (database reload).
    8. REST API endpoints (/portfolios, /bucket, /rebalance, /snapshots, /portfolio).
"""
import pytest
from backend.app.db import db
from backend.app.models.paper_trading import (
    PaperPortfolio,
    PaperPosition,
    PaperSnapshotRecord,
    PaperTrade,
    PortfolioBucketItem,
)
from backend.app.services.trading_pipeline_service import TradingPipelineService


@pytest.fixture
def service(app):
    with app.app_context():
        return TradingPipelineService()


# =========================================================================
# 1. Bucket Validation & Creation Tests
# =========================================================================

def test_validate_bucket_allocations_equal_weight(service):
    """Test equal-weight allocation automatically splits weight evenly."""
    bucket = [
        {"symbol": "RELIANCE.NS"},
        {"symbol": "TCS.NS"},
        {"symbol": "INFY.NS"},
        {"symbol": "SBIN.NS"},
    ]
    validated = service.validate_bucket_allocations(bucket, allocation_method="EQUAL_WEIGHT")
    assert len(validated) == 4
    for item in validated:
        assert item["target_weight"] == 0.25


def test_validate_bucket_allocations_custom_valid(service):
    """Test custom allocation validates and normalizes weights summing to 100%."""
    bucket = [
        {"symbol": "RELIANCE.NS", "target_weight": 0.40},
        {"symbol": "TCS.NS", "target_weight": 0.30},
        {"symbol": "INFY.NS", "target_weight": 0.30},
    ]
    validated = service.validate_bucket_allocations(bucket, allocation_method="CUSTOM")
    assert len(validated) == 3
    total_w = sum(it["target_weight"] for it in validated)
    assert abs(total_w - 1.0) < 1e-5


def test_validate_bucket_allocations_custom_percentages(service):
    """Test user entering percentages (e.g. 50, 50) is normalized to fractions."""
    bucket = [
        {"symbol": "ICICIBANK.NS", "target_pct": 60.0},
        {"symbol": "SBIN.NS", "target_pct": 40.0},
    ]
    validated = service.validate_bucket_allocations(bucket, allocation_method="CUSTOM")
    assert validated[0]["target_weight"] == pytest.approx(0.60)
    assert validated[1]["target_weight"] == pytest.approx(0.40)


def test_validate_bucket_rejects_empty(service):
    """Test empty stock bucket is rejected."""
    with pytest.raises(ValueError, match="Stock bucket cannot be empty"):
        service.validate_bucket_allocations([], allocation_method="EQUAL_WEIGHT")


def test_validate_bucket_rejects_unsupported_symbol(service):
    """Test unsupported symbol is rejected."""
    bucket = [{"symbol": "APPLE.US", "target_weight": 1.0}]
    with pytest.raises(ValueError, match="not supported"):
        service.validate_bucket_allocations(bucket, allocation_method="CUSTOM")


def test_validate_bucket_rejects_duplicate_symbol(service):
    """Test duplicate symbols in bucket are rejected."""
    bucket = [
        {"symbol": "RELIANCE.NS", "target_weight": 0.5},
        {"symbol": "RELIANCE.NS", "target_weight": 0.5},
    ]
    with pytest.raises(ValueError, match="Duplicate symbol"):
        service.validate_bucket_allocations(bucket, allocation_method="CUSTOM")


def test_validate_bucket_rejects_negative_weight(service):
    """Test negative target weight is rejected."""
    bucket = [
        {"symbol": "RELIANCE.NS", "target_weight": 1.2},
        {"symbol": "TCS.NS", "target_weight": -0.2},
    ]
    with pytest.raises(ValueError, match="strictly positive"):
        service.validate_bucket_allocations(bucket, allocation_method="CUSTOM")


def test_validate_bucket_rejects_sum_not_100(service):
    """Test custom weights that do not sum to 100% are rejected."""
    bucket = [
        {"symbol": "RELIANCE.NS", "target_weight": 0.3},
        {"symbol": "TCS.NS", "target_weight": 0.3},
    ]
    with pytest.raises(ValueError, match="must equal 100%"):
        service.validate_bucket_allocations(bucket, allocation_method="CUSTOM")


# =========================================================================
# 2. Portfolio Creation & Model Tests
# =========================================================================

def test_create_portfolio_success(app, service):
    """Test creating a portfolio with valid bucket configuration."""
    with app.app_context():
        bucket = [
            {"symbol": "RELIANCE.NS", "target_weight": 0.5},
            {"symbol": "TCS.NS", "target_weight": 0.5},
        ]
        port = service.create_portfolio(
            name="Tech_Core",
            initial_cash=250000.0,
            bucket=bucket,
            allocation_method="CUSTOM",
            strategy_type="BUY_AND_HOLD",
            description="Core Indian Tech Portfolio",
        )

        assert port.id is not None
        assert port.name == "Tech_Core"
        assert port.initial_cash == 250000.0
        assert port.cash == 250000.0
        assert port.portfolio_value == 250000.0
        assert port.allocation_method == "CUSTOM"
        assert len(port.bucket_items) == 2
        assert len(port.positions) == 0  # Crucial: Bucket does NOT create holdings!


def test_create_portfolio_duplicate_name_rejected(app, service):
    """Test duplicate portfolio name is rejected."""
    with app.app_context():
        bucket = [{"symbol": "INFY.NS"}]
        service.create_portfolio(name="Alpha", initial_cash=100000.0, bucket=bucket)
        with pytest.raises(ValueError, match="already exists"):
            service.create_portfolio(name="Alpha", initial_cash=100000.0, bucket=bucket)


def test_create_portfolio_low_capital_rejected(app, service):
    """Test initial capital below minimum is rejected."""
    with app.app_context():
        bucket = [{"symbol": "INFY.NS"}]
        with pytest.raises(ValueError, match="at least ₹1,000.00"):
            service.create_portfolio(name="Micro", initial_cash=500.0, bucket=bucket)


# =========================================================================
# 3. Portfolio Isolation Tests
# =========================================================================

def test_portfolio_isolation(app, service):
    """Verify that multiple portfolios maintain isolated cash, positions, and trades."""
    with app.app_context():
        p1 = service.create_portfolio(
            name="Portfolio_A",
            initial_cash=100000.0,
            bucket=[{"symbol": "RELIANCE.NS"}],
        )
        p2 = service.create_portfolio(
            name="Portfolio_B",
            initial_cash=500000.0,
            bucket=[{"symbol": "TCS.NS"}],
        )

        # Mutate Portfolio A cash and add position
        p1.cash = 60000.0
        pos1 = PaperPosition(
            portfolio_id=p1.id,
            symbol="RELIANCE.NS",
            quantity=15.0,
            average_entry_price=2500.0,
            current_price=2600.0,
            unrealized_pnl=1500.0,
        )
        db.session.add(pos1)
        db.session.commit()

        # Check Portfolio B remains completely untouched
        refreshed_p2 = service.get_portfolio("Portfolio_B")
        assert refreshed_p2.cash == 500000.0
        assert len(refreshed_p2.positions) == 0

        # Check Portfolio A has its position
        refreshed_p1 = service.get_portfolio("Portfolio_A")
        assert refreshed_p1.cash == 60000.0
        assert len(refreshed_p1.positions) == 1
        assert refreshed_p1.positions[0].symbol == "RELIANCE.NS"


# =========================================================================
# 4. Bucket Modification Tests
# =========================================================================

def test_update_portfolio_bucket(app, service):
    """Test modifying a portfolio's stock selection bucket."""
    with app.app_context():
        p = service.create_portfolio(
            name="Modifiable",
            initial_cash=100000.0,
            bucket=[{"symbol": "RELIANCE.NS"}],
        )
        assert len(p.bucket_items) == 1

        # Update to 3 stocks equal weight
        updated = service.update_portfolio_bucket(
            name_or_id="Modifiable",
            bucket=[{"symbol": "RELIANCE.NS"}, {"symbol": "TCS.NS"}, {"symbol": "INFY.NS"}],
            allocation_method="EQUAL_WEIGHT",
        )
        assert len(updated.bucket_items) == 3
        symbols = [item.symbol for item in updated.bucket_items]
        assert "TCS.NS" in symbols
        assert "INFY.NS" in symbols


# =========================================================================
# 5. REST API Integration Tests
# =========================================================================

def test_api_list_portfolios(client):
    """Test GET /api/v1/trading/portfolios returns list including default."""
    res = client.get("/api/v1/trading/portfolios")
    assert res.status_code == 200
    json_data = res.get_json()
    assert json_data["success"] is True
    assert isinstance(json_data["data"], list)
    names = [p["name"] for p in json_data["data"]]
    assert "default" in names


def test_api_create_portfolio(client):
    """Test POST /api/v1/trading/portfolios creates new user portfolio."""
    payload = {
        "name": "Growth_50",
        "initial_cash": 150000.0,
        "allocation_method": "CUSTOM",
        "strategy_type": "BUY_AND_HOLD",
        "bucket": [
            {"symbol": "RELIANCE.NS", "target_weight": 0.5},
            {"symbol": "ICICIBANK.NS", "target_weight": 0.5},
        ],
    }
    res = client.post("/api/v1/trading/portfolios", json=payload)
    assert res.status_code == 201
    json_data = res.get_json()
    assert json_data["success"] is True
    data = json_data["data"]
    assert data["name"] == "Growth_50"
    assert data["cash"] == 150000.0
    assert len(data["bucket"]) == 2


def test_api_create_portfolio_invalid_allocations(client):
    """Test POST /api/v1/trading/portfolios rejects invalid allocation weights."""
    payload = {
        "name": "Bad_Weights",
        "initial_cash": 100000.0,
        "allocation_method": "CUSTOM",
        "bucket": [
            {"symbol": "RELIANCE.NS", "target_weight": 0.2},
            {"symbol": "TCS.NS", "target_weight": 0.2},
        ],
    }
    res = client.post("/api/v1/trading/portfolios", json=payload)
    assert res.status_code == 400
    json_data = res.get_json()
    assert json_data["success"] is False
    assert "100%" in json_data["error"]["message"]


def test_api_get_portfolio_detail(client):
    """Test GET /api/v1/trading/portfolios/<name>."""
    # First create
    client.post("/api/v1/trading/portfolios", json={
        "name": "Detail_Test",
        "initial_cash": 120000.0,
        "bucket": [{"symbol": "SBIN.NS"}],
    })

    res = client.get("/api/v1/trading/portfolios/Detail_Test")
    assert res.status_code == 200
    data = res.get_json()["data"]
    assert data["name"] == "Detail_Test"
    assert data["cash"] == 120000.0
    assert len(data["bucket"]) == 1


def test_api_delete_portfolio(client):
    """Test DELETE /api/v1/trading/portfolios/<name> (custom allowed, default protected)."""
    # Delete default should fail
    res_def = client.delete("/api/v1/trading/portfolios/default")
    assert res_def.status_code == 400

    # Create and delete custom
    client.post("/api/v1/trading/portfolios", json={
        "name": "To_Delete",
        "initial_cash": 100000.0,
        "bucket": [{"symbol": "INFY.NS"}],
    })
    res_del = client.delete("/api/v1/trading/portfolios/To_Delete")
    assert res_del.status_code == 200
    assert res_del.get_json()["data"]["deleted"] is True

    # Detail now 404
    res_check = client.get("/api/v1/trading/portfolios/To_Delete")
    assert res_check.status_code == 404


def test_api_update_portfolio_bucket(client):
    """Test PUT /api/v1/trading/portfolios/<name>/bucket."""
    client.post("/api/v1/trading/portfolios", json={
        "name": "Bucket_Update_Test",
        "initial_cash": 100000.0,
        "bucket": [{"symbol": "RELIANCE.NS"}],
    })

    # Update to 2 stocks equal weight
    res = client.put("/api/v1/trading/portfolios/Bucket_Update_Test/bucket", json={
        "bucket": [{"symbol": "RELIANCE.NS"}, {"symbol": "TCS.NS"}],
        "allocation_method": "EQUAL_WEIGHT",
    })
    assert res.status_code == 200
    data = res.get_json()["data"]
    assert len(data["bucket"]) == 2
    for b in data["bucket"]:
        assert b["target_weight"] == 0.5


def test_api_rebalance_portfolio_execution(client):
    """Test POST /api/v1/trading/portfolios/<name>/rebalance executes simulated trades."""
    client.post("/api/v1/trading/portfolios", json={
        "name": "Rebal_Test",
        "initial_cash": 100000.0,
        "allocation_method": "EQUAL_WEIGHT",
        "bucket": [{"symbol": "RELIANCE.NS"}, {"symbol": "TCS.NS"}],
    })

    # Trigger rebalance
    res = client.post("/api/v1/trading/portfolios/Rebal_Test/rebalance", json={})
    assert res.status_code == 200
    res_data = res.get_json()["data"]
    assert res_data["success"] is True
    assert res_data["trades_count"] > 0
    assert len(res_data["trades"]) == 2

    # Check that positions are now held
    port_data = res_data["portfolio"]
    assert len(port_data["positions"]) == 2
    assert port_data["cash"] < 100000.0  # Cash was deployed to purchase shares

    # Verify snapshots endpoint returns the recorded snapshot
    res_snap = client.get("/api/v1/trading/portfolios/Rebal_Test/snapshots")
    assert res_snap.status_code == 200
    snapshots = res_snap.get_json()["data"]
    assert len(snapshots) >= 1
    assert snapshots[0]["portfolio_value"] > 0


def test_portfolio_persistence_across_sessions(app, service):
    """Test that portfolios and their bucket configurations persist in SQLite."""
    with app.app_context():
        service.create_portfolio(
            name="Persistent_One",
            initial_cash=75000.0,
            bucket=[{"symbol": "INFY.NS", "target_weight": 1.0}],
            allocation_method="CUSTOM",
        )
        db.session.expire_all()

        # Query in fresh query
        port = PaperPortfolio.query.filter_by(name="Persistent_One").first()
        assert port is not None
        assert port.initial_cash == 75000.0
        assert len(port.bucket_items) == 1
        assert port.bucket_items[0].symbol == "INFY.NS"

