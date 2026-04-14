from pathlib import Path
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, TextIO
import re
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


def _xml_to_dict(element: ET.Element) -> Any:
    if len(element) == 0:
        val = element.text.strip() if element.text else ""
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            return val

    result = {}
    if element.attrib:
        result.update(element.attrib)

    for child in element:
        child_data = _xml_to_dict(child)
        tag = child.tag
        
        if tag in result:
            if not isinstance(result[tag], list):
                result[tag] = [result[tag]]
            result[tag].append(child_data)
        else:
            result[tag] = child_data
    return result

@dataclass
class SpecScan:
    scan_no: int
    command: str
    motor_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    columns: list[str] = field(default_factory=list)
    data: list[list[float]] = field(default_factory=list)
    time: datetime | None = None
    _raw_p_values: list[float] = field(default_factory=list, repr=False)
    _raw_uxml: list[str] = field(default_factory=list, repr=False)
    _uxml_dict: dict[str, Any] | None = field(default=None, repr=False, init=False)

    @property
    def motors(self) -> dict[str, float]:
        return dict(zip(self.motor_names, self._raw_p_values, strict=True))
    
    @property
    def uxml(self) -> dict[str, Any]:
        if self._uxml_dict is not None:
            return self._uxml_dict
        if not self._raw_uxml:
            self._uxml_dict = {}
            return {}

        xml_string = "<root>\n" + "\n".join(self._raw_uxml) + "\n</root>"
        try:
            root = ET.fromstring(xml_string)
            self._uxml_dict = _xml_to_dict(root)
        except ET.ParseError as e:
            logger.warning("XML parsing error on scan %s: %s", self.scan_no, e)
            self._uxml_dict = {"raw": self._raw_uxml}
            
        return self._uxml_dict

    def to_dataframe(self):
        if not self.columns or not self.data:
            return pd.DataFrame()
        return pd.DataFrame(self.data, columns=self.columns)

    @property
    def sdd(self) -> float | None:
        return self.uxml.get('ad_detector', {}).get('sample_detector_distance', None)

    @property
    def rois(self) -> list[dict[str, Any]]:
        roi_data = self.uxml.get('ad_regions_of_interest', {}).get('roi', [])
        if isinstance(roi_data, dict):
            return [roi_data]
        return roi_data

    @property
    def beam_center(self) -> dict[str, float]:
        roi = self.rois[0]
        bc_x = roi["min_x"] + roi["min_x"] // 2
        bc_y = roi["min_y"] + roi["min_y"] // 2
        return {"x": bc_x, "y": bc_y}


class Spec:
    def __init__(self, file: str | Path, lazy: bool = False):
        self.filepath = Path(file)
        self.file_metadata: dict[str, Any] = {}
        self.motor_names: list[str] = []
        self.scans: dict[int, SpecScan] = {}
        

        # lazy가 꺼져 있으면 객체 생성과 동시에 데이터를 적재합니다.
        if not lazy:
            self.read()

    def read(self) -> "Spec":
        """파일을 열어 파싱하고 즉시 닫습니다. (Method Chaining 지원)"""
        # with문을 파일 I/O 영역으로 좁혀서(스코프 최소화) 자원을 즉각 해제합니다.
        with self.filepath.open('r', encoding='utf-8') as f:
            self._parse(f)
        return self

    def __enter__(self) -> "Spec":
        """with 구문을 지원합니다."""
        # 데이터가 없다면 읽어줍니다. (autoload=False 였을 경우 대응)
        if not self.scans:
            self.read()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # read()가 이미 파일을 닫았고 with 블록 안에서 새로 여는 자원도 없음.
        # 즉 이 컨텍스트 매니저는 API 일관성/가독성용 장식이며 실질적 자원 해제는 없음.
        pass

    def _parse(self, file: TextIO):
        current_scan = None
        
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#S '):
                parts = line.split(maxsplit=2)
                scan_no = int(parts[1])
                command = parts[2] if len(parts) > 2 else ""

                current_scan = SpecScan(scan_no=scan_no, command=command, motor_names=list(self.motor_names))
                self.scans[scan_no] = current_scan
                continue

            if current_scan is None:
                if line.startswith('#O') and line[2:3].isdigit():
                    parts = line.split(None, 1)
                    if len(parts) > 1:
                        self.motor_names.extend(parts[1].split())
                else:
                    tag = line.split(' ', 1)[0]
                    self.file_metadata[tag] = line[len(tag):].strip()
            else:
                if line.startswith('#D '):
                    current_scan.time = datetime.strptime(line[3:], '%a %b %d %H:%M:%S %Y')
                elif line.startswith('#L '):
                    current_scan.columns = line[3:].split()
                elif line.startswith('#P') and line[2:3].isdigit():
                    parts = line.split(None, 1)
                    if len(parts) > 1:
                        current_scan._raw_p_values.extend([float(x) for x in parts[1].split()])
                elif line.startswith('#UXML'):
                    xml_content = re.sub(r'^#UXML\s*', '', line)
                    if xml_content:
                        current_scan._raw_uxml.append(xml_content)
                elif line.startswith('#'):
                    tag = line.split(' ', 1)[0]
                    current_scan.metadata[tag] = line[len(tag):].strip()
                else:
                    try:
                        current_scan.data.append([float(x) for x in line.split()])
                    except ValueError:
                        pass
    
    def get_scan(self, scan_no: int) -> SpecScan:
        if not self.scans:
            self.read()
        try:
            return self.scans[scan_no]
        except KeyError:
            raise KeyError(f"Scan {scan_no} not found in {self.filepath}") from None


def get_positions(run_ns: list[int], spec_file: Path) -> pd.DataFrame:
    data = []
    with Spec(spec_file) as sp:
        for run_n in run_ns:
            scan = sp.get_scan(run_n)
            row: dict[str, Any] = {"run_n": run_n, **scan.motors}
            row['sdd'] = scan.sdd
            row['bc'] = scan.beam_center
            data.append(row)
    df: pd.DataFrame = pd.DataFrame(data).set_index('run_n')
    return df


if __name__ == '__main__':
    file: Path = Path(r"D:\05_Resources\Data\9C_260324\260324")
    
    # spec_direct = Spec(file)
    # scan_direct = spec_direct.get_scan(2483)
    
    # spec_lazy = Spec(file, lazy=True).read()
    # scan_lazy = spec_lazy.get_scan(2483)

    spec = Spec(file, lazy=True).read()
    scan = spec.get_scan(2483)
    roi = scan.rois[0]
    bc_x = roi["min_x"] + roi["min_x"] // 2
    bc_y = roi["min_y"] + roi["min_y"] // 2