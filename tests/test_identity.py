from coreason_identity.types import SecretStr


def test_secret_str_methods() -> None:
    s = SecretStr("my-secret")
    assert s.get_secret_value() == "my-secret"
    assert str(s) == "**********"
    assert repr(s) == "SecretStr('**********')"
