"""Compatibility helpers for old msgpack-rpc-python on new msgpack.

msgpack-rpc-python 0.4.1 still calls msgpack.Packer/Unpacker with the
removed ``encoding=`` argument.  ProjectAirSim's Python dependencies may pull
in msgpack 1.x, so we patch only the socket constructor used by msgpackrpc.
"""


def patch_msgpackrpc_encoding() -> None:
    """Make msgpack-rpc-python work with msgpack 1.x.

    旧版 msgpack-rpc-python 写死了 ``encoding=`` 参数；新版 msgpack
    改成了 ``raw=``。这里不改第三方库文件，只在当前进程里替换
    BaseSocket.__init__，让 RPC server 和 client 都能正常收发消息。
    """
    try:
        import msgpack
        import msgpackrpc.transport.tcp as tcp
    except Exception:
        return

    try:
        msgpack.Packer(encoding="utf-8")
        msgpack.Unpacker(encoding=None)
        return
    except TypeError as error:
        if "encoding" not in str(error):
            return

    if getattr(tcp.BaseSocket, "_projectairsim_encoding_patch", False):
        return

    def _base_socket_init(self, stream, encodings):
        pack_encoding, unpack_encoding = encodings or ("utf-8", None)
        del pack_encoding

        self._stream = stream
        self._packer = msgpack.Packer(
            default=lambda value: value.to_msgpack(),
            use_bin_type=False,
        )
        self._unpacker = msgpack.Unpacker(
            raw=unpack_encoding is None,
            strict_map_key=False,
        )

    tcp.BaseSocket.__init__ = _base_socket_init
    tcp.BaseSocket._projectairsim_encoding_patch = True
