from typing import cast, Dict


SEND_PARAMETERS = "send_parameters"


class ClientSideMetadata:
    def __init__(self, message_type):
        var_dict = {
            "_message_type": message_type
        }
        self.__dict__.update(var_dict)

    @property
    def message_type(self) -> str:
        """A string that encodes the action to be executed on the receiving end."""
        return cast(str, self.__dict__["_message_type"])

    @message_type.setter
    def message_type(self, value: str) -> None:
        """Set message_type."""
        self.__dict__["_message_type"] = value

    def __repr__(self) -> str:
        """Return a string representation of this instance."""
        view = ", ".join([f"{k.lstrip('_')}={v!r}" for k, v in self.__dict__.items()])
        return f"{self.__class__.__qualname__}({view})"


class ClientSideMessage:
    def __init__(self, content: Dict, metadata: ClientSideMetadata) -> None:
        var_dict = {
            "_content": content,
            "_metadata": metadata
        }
        self.__dict__.update(var_dict)

    @property
    def content(self) -> Dict:
        """The content of this message."""
        if self.__dict__["_content"] is None:
            raise ValueError(
                "Message content is None. Use <message>.has_content() "
                "to check if a message has content."
            )
        return cast(Dict, self.__dict__["_content"])

    @content.setter
    def content(self, value: Dict) -> None:
        """Set content."""
        if self.__dict__["_content"] is not None:
            self.__dict__["_content"] = value
        else:
            raise ValueError("A message with an error set cannot have content.")

    @property
    def metadata(self) -> ClientSideMetadata:
        """A dataclass including information about the message to be executed."""
        return cast(ClientSideMetadata, self.__dict__["_metadata"])
