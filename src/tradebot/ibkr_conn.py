"""Shared Interactive Brokers connection (pip install 'tradebot[ibkr]').

Requires TWS or IB Gateway running locally with the API enabled.
Ports: 7497 TWS paper, 7496 TWS live, 4002 Gateway paper, 4001 Gateway live.
"""

from __future__ import annotations

LIVE_PORTS = {7496, 4001}
_conn = None


def get_ib(host: str, port: int, client_id: int, allow_live: bool = False):
    global _conn
    if port in LIVE_PORTS and not allow_live:
        raise PermissionError(f"Port {port} is an IBKR LIVE port but live trading is not enabled.")
    if _conn is not None and _conn.isConnected():
        return _conn
    from ib_async import IB

    ib = IB()
    ib.connect(host, port, clientId=client_id)
    _conn = ib
    return ib
