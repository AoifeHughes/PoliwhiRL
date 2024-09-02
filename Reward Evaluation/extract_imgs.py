# -*- coding: utf-8 -*-
import sqlite3
import os
from PIL import Image
import io


def extract_number(loc_string):
    return "".join(filter(str.isdigit, loc_string))


def get_unique_map_num_locs():
    conn = sqlite3.connect("memory_data.db")
    cursor = conn.cursor()
    cursor.execute(
        """
    SELECT DISTINCT map_num_loc
    FROM memory_data
    ORDER BY map_num_loc
    """
    )
    unique_locs = cursor.fetchall()
    conn.close()
    return [extract_number(loc[0]) for loc in unique_locs if extract_number(loc[0])]


def extract_images_by_map_num_loc(map_num_loc):
    conn = sqlite3.connect("memory_data.db")
    cursor = conn.cursor()
    folder_name = f"extract/map_num_loc_{map_num_loc}"
    os.makedirs(folder_name, exist_ok=True)
    cursor.execute(
        """
        SELECT id, image, X, Y, location
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY X, Y, location ORDER BY id) as rn
            FROM memory_data
            WHERE map_num_loc LIKE ?
        )
        WHERE rn = 1
        """,
        (f"%{map_num_loc}%",),
    )
    results = cursor.fetchall()
    print(f"Found {len(results)} unique images for map_num_loc {map_num_loc}")
    for row_id, image_data, x, y, location in results:
        image = Image.open(io.BytesIO(image_data))
        image_path = os.path.join(
            folder_name,
            f"x_{x}_y_{y}_location_{location}_map_num_loc_{map_num_loc}.png",
        )
        count = 1
        while os.path.exists(image_path):
            image_path = os.path.join(
                folder_name,
                f"x_{y}_y_{x}_location_{location}_map_num_loc_{map_num_loc}_{count}.png",
            )
            count += 1
        image.save(image_path)
        print(f"Saved image {row_id} to {image_path}")
    conn.close()
    print(
        f"Extraction complete for map_num_loc {map_num_loc}. Images saved in folder: {folder_name}"
    )


def extract_all_maps():
    unique_locs = get_unique_map_num_locs()
    print(f"Found {len(unique_locs)} unique map locations")
    for loc in unique_locs:
        extract_images_by_map_num_loc(loc)


if __name__ == "__main__":
    extract_all_maps()
