from __future__ import annotations

from dataclasses import dataclass

import base64
import io
import mujoco
from pathlib import Path

def mjspec_to_string(
    spec: mujoco.MjSpec,
    source_xml: str | Path,
) -> str:
    root = Path(source_xml).expanduser().resolve().parent
    compiler = spec.compiler

    def absolute(path: str) -> str:
        if not path:
            return ""
        p = Path(path)
        return str(p if p.is_absolute() else (root / p).resolve())

    compiler.meshdir = absolute(compiler.meshdir)
    compiler.texturedir = absolute(compiler.texturedir)

    # validate spec
    spec.compile()

    archive = io.BytesIO()
    spec.to_zip(archive)

    return base64.b64encode(archive.getvalue()).decode("ascii")


def mjspec_from_string(payload: str) -> mujoco.MjSpec:
    archive_bytes = base64.b64decode(payload.encode("ascii"))
    archive = io.BytesIO(archive_bytes)

    return mujoco.MjSpec.from_zip(archive)

@dataclass(frozen=True)
class MyoModelNames:
    actuator_names: list[str]
    tendon_actuator_names: list[str]
    non_tendon_actuator_names: list[str]

    tendon_names: list[str]

    joint_names: list[str]
    dependent_joint_names: list[str]
    independent_joint_names: list[str]

def myo_get_model_names(model: mujoco.MjModel) -> MyoModelNames:
    actuator_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(model.nu)
    ]

    tendon_actuator_names = [
        actuator_names[i]
        for i in range(model.nu)
        if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_TENDON
    ]

    non_tendon_actuator_names = [
        actuator_names[i]
        for i in range(model.nu)
        if model.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_TENDON
    ]

    tendon_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, i)
        for i in range(model.ntendon)
    ]

    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]

    dependent_joint_ids = set()

    for eq_id in range(model.neq):
        if not model.eq_active0[eq_id]:
            continue

        if model.eq_type[eq_id] != mujoco.mjtEq.mjEQ_JOINT:
            continue

        joint_id = int(model.eq_obj1id[eq_id])
        if 0 <= joint_id < model.njnt:
            dependent_joint_ids.add(joint_id)

    dependent_joint_names = [
        joint_names[i]
        for i in range(model.njnt)
        if i in dependent_joint_ids
    ]

    independent_joint_names = [
        joint_names[i]
        for i in range(model.njnt)
        if i not in dependent_joint_ids
    ]

    return MyoModelNames(
        actuator_names=actuator_names,
        tendon_actuator_names=tendon_actuator_names,
        non_tendon_actuator_names=non_tendon_actuator_names,
        tendon_names=tendon_names,
        joint_names=joint_names,
        dependent_joint_names=dependent_joint_names,
        independent_joint_names=independent_joint_names,
    )