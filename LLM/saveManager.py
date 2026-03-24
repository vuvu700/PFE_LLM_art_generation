"""code repris d'un projet perso"""

import os
from pathlib import Path
import subprocess

from holo import patternValidation
from holo.__typing import PartialyFinalClass, assertIsinstance
from holo.prettyFormats import PrettyfyClass
from holo.files import get_subdirectorys


class SavedAiTree(PrettyfyClass, PartialyFinalClass):
    """handle the detection of folders of version in \
    the directory where an AI is saved"""

    __slots__ = (
        "aiDirectory",
        "__maxVersion",
        "__versionsFolders",
    )
    __finals__ = {
        "aiDirectory",
        "__versionsFolders",
    }

    __ALL_VERSIONS_DIRECTORY = "versions"

    __VERSION_PREFIX = "version"
    __VERSION_NB_DIGITS = 4
    __VERSION_FOLDER_PATTERN = f"{__VERSION_PREFIX}_<v:d>_<name>"

    def __init__(self, aiFolder: "Path") -> None:
        self.aiDirectory: Path = aiFolder
        """the dir of the AI"""
        self.__versionsFolders: "dict[int, Path]" = {}
        """all the paths to the folders storing the versions"""
        self.__maxVersion: "int|None" = None
        """None -> no version detected"""
        self.update()

    def update(self) -> None:
        # ensure the main dir exist
        if not self.aiDirectory.exists():
            self.aiDirectory.mkdir(parents=False, exist_ok=False)
        # ensure the versions dir exist
        allVersionsDir = self.allVersionsDir
        if allVersionsDir.exists() is False:
            allVersionsDir.mkdir(parents=False, exist_ok=False)
        # reset the scaned infos
        self.__versionsFolders.clear()
        self.__maxVersion = None
        # scan the versions directory
        for dirName in get_subdirectorys(allVersionsDir):
            matched, extracted = patternValidation(
                dirName, self.__VERSION_FOLDER_PATTERN
            )
            if matched is False:
                continue  # => this isn't a version folder
            # => this is a version dir
            versionID = assertIsinstance(int, extracted["v"])
            versionName = assertIsinstance(str, extracted["name"])
            self.__versionsFolders[versionID] = allVersionsDir.joinpath(dirName)
            # update the maxVersion known
            if (self.__maxVersion is None) or (versionID > self.__maxVersion):
                self.__maxVersion = versionID

    @property
    def allVersionsDir(self) -> Path:
        """the path to the versions directory"""
        return self.aiDirectory.joinpath(self.__ALL_VERSIONS_DIRECTORY)

    @property
    def currentLatestVersion(self) -> "int|None":
        """the ID of the last version known"""
        return self.__maxVersion

    @property
    def currentNextVersion(self) -> int:
        """the ID of the next version to create"""
        currMax = self.currentLatestVersion
        if currMax is None:
            return 1
        else:
            return currMax + 1

    def getVersionDirectory(self, versionID: int) -> Path:
        if versionID not in self.__versionsFolders.keys():
            raise KeyError(f"there is no known version: {versionID}")
        return Path(self.__versionsFolders[versionID])

    def __getNextVersionFolder(self, versionName: str) -> Path:
        """return the path to the "next" version using the given name"""
        newDir = self.allVersionsDir.joinpath(
            f"{self.__VERSION_PREFIX}_"
            f"{self.currentNextVersion:0{self.__VERSION_NB_DIGITS}d}_{versionName}"
        )
        if newDir.parent.as_posix() != self.allVersionsDir.as_posix():
            raise ValueError(f"the version name: {versionName!r} isn't valid")
        return newDir

    def createNewVersionFolder(self, versionName: str) -> Path:
        """create the new folder to save the AI and update the `latest` link"""
        newVersionFolder = self.__getNextVersionFolder(versionName)
        newVersionFolder.mkdir(parents=False, exist_ok=False)  # expect a new dir
        # => new version created
        versionID = self.currentNextVersion
        self.__versionsFolders[versionID] = newVersionFolder
        self.__maxVersion = versionID
        return newVersionFolder
