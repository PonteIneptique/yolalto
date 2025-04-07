"""
Requirements: `shapely`, `ultralytics`, `lxml`, `click`
"""
from ultralytics import YOLO
from lxml import etree
import itertools
from typing import List, Dict, Tuple
import uuid
import os
import hashlib
import click
from pathlib import Path
import math


def sort_by_top_left_distance(bbox_objects: List[Dict]) -> List[Dict]:
    """
    Sorts regions based on the Euclidean distance from the top-left corner (0, 0) of the page.

    Args:
        bbox_objects (List[Dict]): A list of dictionaries, each with a "bbox" key as [x1, y1, x2, y2].

    Returns:
        List[Dict]: The list of regions sorted by distance of top-left corner of bbox to (0, 0).
    """
    def distance_from_origin(x1, y1):
        return math.hypot(x1, y1)

    return sorted(bbox_objects, key=lambda obj: distance_from_origin(*obj["bbox"][:2]))


def load_model(model_path: str) -> YOLO:
    """Load the YOLO model."""
    return YOLO(model_path)


def create_tag_id(label: str, prefix: str) -> str:
    """
    Create a stable hash-based tag ID with a given prefix.
    """
    # Hash the label to get a stable numeric suffix
    h = hashlib.sha1(label.encode()).hexdigest()[:6]
    numeric_part = int(h, 16) % 100000
    return f"{prefix}{numeric_part}"

    
def run_inference(model: YOLO, images_path: List[str]) -> List[Tuple[Dict, Tuple[int, int]]]:
    """Run YOLO on an image and return parsed predictions."""
    results = model(images_path)
    out = []
    for result in results:
        detections = []
        h, w = result.orig_shape
        for box in result.boxes:
            label = result.names[int(box.cls)]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({'label': label, 'bbox': [x1, y1, x2, y2]})
        out.append((detections, (w, h)))
    return out


def relative_intersection(line_bbox: List[float], zone_bbox: List[float]) -> float:
    """Compute IoU between two bounding boxes."""
    xA = max(line_bbox[0], zone_bbox[0])
    yA = max(line_bbox[1], zone_bbox[1])
    xB = min(line_bbox[2], zone_bbox[2])
    yB = min(line_bbox[3], zone_bbox[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    area1 = (line_bbox[2] - line_bbox[0]) * (line_bbox[3] - line_bbox[1])
    if area1 == 0.0:
        return 0
    return inter_area / area1


def iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    if inter_area == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = areaA + areaB - inter_area

    return inter_area / union_area


def filter_lines(lines: List[Dict]) -> List[int]:
    bboxes = {idx: line["bbox"] for idx, line in enumerate(lines)}
    to_delete = set()
    checked_pairs = set()

    for (id1, box1), (id2, box2) in itertools.combinations(bboxes.items(), 2):
        pair_key = tuple(sorted((id1, id2)))
        if pair_key in checked_pairs:
            continue
        checked_pairs.add(pair_key)
        score = iou(box1, box2)
        if score > 0.5:
            delete_id = id1 if abs(box1[0]-box1[2]) > abs(box2[0]-box2[2]) else id2
            to_delete.add(delete_id)

    return list(to_delete)


def assign_lines_to_zones(detections: List[Dict]) -> Dict[str, List[Dict]]:
    """Assign lines to zones using maximum IoU."""
    zones = [d for d in detections if d["label"].endswith("Zone")]
    lines = [d for d in detections if d["label"].endswith("Line")]

    assignments = {str(i): [] for i in range(len(zones))}

    for line in lines:
        best_idx = -1
        best_intersection = 0
        for idx, zone in enumerate(zones):
            current_intersection = relative_intersection(line["bbox"], zone["bbox"])
            if current_intersection > best_intersection:
                best_intersection = current_intersection
                best_idx = idx
        if best_idx != -1:
            assignments[str(best_idx)].append(line)

    return {"zones": zones, "assignments": assignments}


def bbox_to_polygon(bbox: List[float]) -> str:
    """Convert a bbox to a 4-point polygon string for ALTO."""
    x1, y1, x2, y2 = map(int, bbox)
    return f"{x1} {y1} {x2} {y1} {x2} {y2} {x1} {y2}"


def bbox_baseline(bbox: List[float], num_points: int = 10) -> str:
    """
    Generate a baseline from left to right, in the middle of the bbox,
    as a space-separated string of x,y points.
    """
    # Usable if moving to OBbox
    # x1, y1, x2, y2 = map(int, bbox)
    # center_y = (y1 + y2) // 2
    # xs = np.linspace(x1, x2, num_points, endpoint=True).astype(int)
    # points = [f"{x},{center_y}" for x in xs]
    # return " ".join(points)
    x1, y1, x2, y2 = map(int, bbox)
    center_y = (y1 + y2) // 2
    return f"{x1} {center_y} {x2} {center_y}"


def create_alto(zones: List[Dict], assignments: Dict[str, List[Dict]], image_path: str, wh: Tuple[int, int]) -> etree._Element:
    """Generate ALTO XML from zones and lines."""

    # Build tag registry for all unique labels
    tag_registry = {}
    tags = set()
    
    for zone in zones:
        label = zone["label"]
        if label not in tag_registry:
            tag_id = create_tag_id(label, "BT")
            tag_registry[label] = tag_id
            tags.add((tag_id, label, "block type"))
    
    for line_list in assignments.values():
        for line in line_list:
            label = line["label"]
            if label not in tag_registry:
                tag_id = create_tag_id(label, "LT")
                tag_registry[label] = tag_id
                tags.add((tag_id, label, "line type"))
    
    NSMAP = {
        None: "http://www.loc.gov/standards/alto/ns-v4#",
        "xsi": "http://www.w3.org/2001/XMLSchema-instance"
    }

    alto = etree.Element(
        "alto",
        nsmap=NSMAP,
        attrib={
            "{http://www.w3.org/2001/XMLSchema-instance}schemaLocation":
                "http://www.loc.gov/standards/alto/ns-v4# http://www.loc.gov/standards/alto/v4/alto-4-2.xsd"
        }
    )

    description = etree.SubElement(alto, "Description")
    etree.SubElement(description, "MeasurementUnit").text = "pixel"

    source_info = etree.SubElement(description, "sourceImageInformation")
    filename = os.path.basename(image_path)
    etree.SubElement(source_info, "fileName").text = filename
    etree.SubElement(source_info, "fileIdentifier").text = filename

    tags_elem = etree.SubElement(alto, "Tags")
    for tag_id, label, tag_type in sorted(tags):
        etree.SubElement(tags_elem, "OtherTag", ID=tag_id, LABEL=label, DESCRIPTION=f"{tag_type} {label}")

    layout = etree.SubElement(alto, "Layout")
    page = etree.SubElement(layout, "Page", ID="page1", PHYSICAL_IMG_NR="1", HEIGHT=str(wh[1]), WIDTH=str(wh[0]))

    print_space = etree.SubElement(page, "PrintSpace")

    # Sort zones top-to-bottom then left-to-right
    # zone_order = sorted(enumerate(zones), key=lambda z: (z[1]['bbox'][1], z[1]['bbox'][0]))
    zones = [{"idx": idx, **zone} for (idx, zone) in enumerate(zones)]
    zones = sort_by_top_left_distance(zones)
    # return
    for zone in zones:
        x1, y1, x2, y2 = map(int, zone['bbox'])
        tb = etree.SubElement(print_space, "TextBlock", ID=f"zone_{zone['idx']}", HPOS=str(x1), VPOS=str(y1),
                              WIDTH=str(x2 - x1), HEIGHT=str(y2 - y1),TAGREFS=tag_registry[zone["label"]])
        shape = etree.SubElement(tb, "Shape")
        etree.SubElement(shape, "Polygon", POINTS=bbox_to_polygon(zone["bbox"]))
        # for line in sorted(assignments[str(idx)], key=lambda l: (l["bbox"][1], l["bbox"][0])):
        for line in sort_by_top_left_distance(assignments[str(zone["idx"])]):
            lx1, ly1, lx2, ly2 = map(int, line["bbox"])
            tl = etree.SubElement(
                tb,
                "TextLine",
                ID=str(uuid.uuid4()),
                HPOS=str(lx1),
                VPOS=str(ly1),
                WIDTH=str(lx2 - lx1),
                HEIGHT=str(ly2 - ly1),
                BASELINE=bbox_baseline(line["bbox"]),
                TAGREFS=tag_registry[line["label"]]
            )
            # baseline = etree.SubElement(tl, "Baseline", )
            shape = etree.SubElement(tl, "Shape")
            etree.SubElement(shape, "Polygon", POINTS=bbox_to_polygon(line["bbox"]))

    return alto


def save_alto_xml(alto_element: etree._Element, output_path: str):
    """Write ALTO XML to file."""
    tree = etree.ElementTree(alto_element)
    tree.write(output_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")


@click.command()
@click.argument('image_paths', type=click.Path(exists=True, path_type=Path), nargs=-1)
@click.argument('model_path', type=click.Path(exists=True, path_type=Path))
@click.option('-b', '--batch-size', default=4, help="Batch size for processing images.")
@click.option("-d", "--device", default=None, help="Device to use")
def cli(image_paths: tuple[Path], model_path: Path, batch_size: int = 4, device: str = None):
    """Generate ALTO XML from multiple images using a YOLOv8 model."""
    model = load_model(str(model_path))
    if device:
        model.to(device)
    if batch_size:
        image_paths = list(image_paths)  # Convert to list for easier batching
        num_batches = len(image_paths) // batch_size + (1 if len(image_paths) % batch_size else 0)


    for batch_idx in range(num_batches):
        batch = image_paths[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        batch_detections = run_inference(model, batch)
        for image_path, (detections, wh) in zip(batch, batch_detections):
            output_path = Path(image_path).with_suffix('.xml')
            print(f"Processing {image_path} and saving to {output_path}")
            result = assign_lines_to_zones(detections)
            alto = create_alto(result["zones"], result["assignments"], os.path.basename(str(image_path)), wh)
            save_alto_xml(alto, str(output_path))

cli()