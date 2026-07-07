import pathlib
from typing import TextIO
from toktagger.api.config import Settings


def create_default_toml_file(file_path: pathlib.Path | str):
    """
    Create a TOML file containing all possible settings options, with their defaults and descriptions (if available)

    Parameters
    ----------
    file_path : pathlib.Path | str
        The path to create the TOML file at
    """

    schema = Settings().model_json_schema()

    def _walk_schema(file: TextIO, schema: dict[str, dict], heading: str):
        """Walk through schema recursively and create sections of TOML file.

        Parameters
        ----------
        file: TextIO
            The file object to write to
        schema: dict[str | dict]
            The schema (or subsection of a schema) to use to write the TOML
        heading: str
            The heading of the current section in the TOML file
        """

        if heading:
            file.write(f"[{heading}]\n")
        for key, values in schema.items():
            # If 'properties' present, need to create a new subsection, as this is a BaseModel
            if properties := values.get("properties"):
                _walk_schema(
                    file,
                    properties,
                    f"{heading.lower()}.{key.lower()}" if heading else key.lower(),
                )
            else:
                # Otherwise we are in the lowest section applicable, so write lines
                if desc := values.get("description"):
                    file.write(f"# {desc}\n")
                default = values.get("default", "<your value here>")
                if type(default) is str:
                    default = f"'{default}'"
                if type(default) is bool:
                    default = "true" if default is True else "false"

                file.write(f"# {key} = {default}\n\n")
        file.write("\n")

    with open(file_path, "w") as out_file:
        out_file.writelines(
            [
                "# Example TokTagger Configuration File. \n",
                "# All settings are optional, and default values are indicated below. \n",
                "# To override a setting, uncomment the line and provide your value.\n\n",
            ]
        )
        _walk_schema(out_file, schema["$defs"], "")


if __name__ == "__main__":
    create_default_toml_file("toktagger.example.toml")
