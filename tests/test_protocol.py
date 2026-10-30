import base64
import json

import pytest

from shared.protocol import (
    ProtocolError,
    SecureChannel,
    b64d,
    b64e,
    make_client_proof,
    make_server_proof,
    new_nonce,
    verify_proof,
)


def make_pair():
    psk = bytes(range(32))
    client_nonce = new_nonce()
    server_nonce = new_nonce()
    session_id = "session-1"
    return (
        psk,
        client_nonce,
        server_nonce,
        session_id,
        SecureChannel.from_psk(
            psk, client_nonce, server_nonce, session_id, role="client"
        ),
        SecureChannel.from_psk(
            psk, client_nonce, server_nonce, session_id, role="server"
        ),
    )


def test_roundtrip_both_directions():
    _, _, _, _, client, server = make_pair()
    envelope = client.encrypt_json("job", {"prompt": "a cat", "steps": 28})
    msg_type, payload = server.decrypt_json(envelope)
    assert msg_type == "job"
    assert payload == {"prompt": "a cat", "steps": 28}

    envelope = server.encrypt_json("progress", {"step": 3, "total": 28})
    msg_type, payload = client.decrypt_json(envelope)
    assert msg_type == "progress"
    assert payload == {"step": 3, "total": 28}


def test_tampered_ciphertext_is_rejected():
    _, _, _, _, client, server = make_pair()
    envelope = client.encrypt_json("job", {"prompt": "a cat"})
    raw = bytearray(b64d(envelope["ct"]))
    raw[-1] ^= 1
    envelope["ct"] = b64e(bytes(raw))
    with pytest.raises(ProtocolError):
        server.decrypt_json(envelope)


def test_replayed_sequence_is_rejected():
    _, _, _, _, client, server = make_pair()
    envelope = client.encrypt_json("progress", {"step": 1})
    server.decrypt_json(envelope)
    with pytest.raises(ProtocolError):
        server.decrypt_json(envelope)


def test_wrong_key_is_rejected():
    psk, client_nonce, server_nonce, session_id, client, _ = make_pair()
    other = SecureChannel.from_psk(
        bytes(reversed(psk)), client_nonce, server_nonce, session_id, role="server"
    )
    envelope = client.encrypt_json("job", {"prompt": "a cat"})
    with pytest.raises(ProtocolError):
        other.decrypt_json(envelope)


def test_session_mismatch_is_rejected():
    _, _, _, _, client, server = make_pair()
    envelope = client.encrypt_json("job", {"prompt": "a cat"})
    envelope["sid"] = "another-session"
    with pytest.raises(ProtocolError):
        server.decrypt_json(envelope)


def test_handshake_proofs_are_mutual_and_key_bound():
    psk = bytes(range(32))
    client_nonce = new_nonce()
    server_nonce = new_nonce()
    session_id = "session-1"
    client_proof = make_client_proof(
        psk, "gpu-01", "k1", client_nonce, server_nonce, session_id
    )
    server_proof = make_server_proof(
        psk, "gpu-01", "k1", client_nonce, server_nonce, session_id
    )
    assert verify_proof(client_proof, client_proof)
    assert verify_proof(server_proof, server_proof)
    assert not verify_proof(client_proof, server_proof)
    assert not verify_proof(
        client_proof,
        make_client_proof(
            bytes(reversed(psk)),
            "gpu-01",
            "k1",
            client_nonce,
            server_nonce,
            session_id,
        ),
    )


def test_worker_key_store_parses_rotation_entries(tmp_path):
    from app.worker_auth import WorkerKeyStore

    key = base64.b64encode(bytes(range(32))).decode()
    path = tmp_path / "workers.json"
    path.write_text(
        json.dumps(
            {
                "workers": [
                    {
                        "worker_id": "gpu-01",
                        "key_id": "old",
                        "key": key,
                        "allow_instances": True,
                    },
                    {"worker_id": "gpu-01", "key_id": "new", "key": key},
                ]
            }
        )
    )
    store = WorkerKeyStore.from_file(path)
    assert store.get("gpu-01", "old") == bytes(range(32))
    assert store.get("gpu-01", "new") == bytes(range(32))
    assert store.get("gpu-01-1", "old") == bytes(range(32))
    assert store.get("gpu-01-12", "old") == bytes(range(32))
    assert store.get("gpu-01-1", "new") is None
    assert store.get("gpu-01-extra", "old") is None
    assert store.get("gpu-02", "new") is None
