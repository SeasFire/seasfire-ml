import logging
import os
import numpy as np
import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class GraphBuilder:
    def __init__(
        self,
        cube,
        cube_resolution,
        input_vars, 
        oci_input_vars,
        target_var,
        output_folder,
        radius,
        timeseries_weeks,
        target_count,
        global_scale_factor,
    ):
        self._input_vars = input_vars
        self._oci_input_vars = oci_input_vars
        self._target_var = target_var

        self._cube = cube
        if cube_resolution == "25km":
            self._sp_res = 0.25
            self._lat_min = -89.875
            self._lat_max = 89.875
            self._lon_min = -179.875
            self._lon_max = 179.875
        elif cube_resolution == "100km":
            self._sp_res = 1
            self._lat_min = -89.5
            self._lat_max = 89.5
            self._lon_min = -179.5
            self._lon_max = 179.5
        else: 
            raise ValueError("Invalid cube resolution")

        # create output cache folder
        self._cache_folder = os.path.join(output_folder, "cache")
        for folder in [self._cache_folder]:
            logger.info("Creating cache folder {}".format(folder))
            if not os.path.exists(folder):
                os.makedirs(folder)

        self._radius = radius
        self._global_scale_factor = global_scale_factor
        self._timeseries_weeks = timeseries_weeks
        self._target_count = target_count


    def _create_local_vertices(self, lat, lon, time, radius):
        # find time in time coords
        time_idx = np.where(self._cube["time"] == time)[0][0]
        time_slice = slice(
            time_idx - self._timeseries_weeks + 1, time_idx + 1
        )

        input_vars = self._input_vars
        if len(self._oci_input_vars) > 0:
            input_vars += self._oci_input_vars
        points_input_vars = self._cube[input_vars].isel(time=time_slice).load()

        timeseries_len = len(points_input_vars.coords["time"])
        if timeseries_len != self._timeseries_weeks:
            logger.warning(
                "Invalid time series length {} != {}".format(
                    timeseries_len, self._timeseries_weeks
                )
            )
            raise ValueError("Invalid time series length")

        lat_coords = self._create_lat_coords(lat, radius, True)
        lon_coords = self._create_lon_coords(lon, radius, True)

        # Create list of vertices
        vertices = []
        vertices_idx = {}
        for lat in lat_coords:
            for lon in lon_coords:
                cur_vertex = (lat, lon)
                vertices_idx[cur_vertex] = len(vertices)
                vertices.append(cur_vertex)

        # Create vertex feature tensors
        vertices_input_vars = points_input_vars.stack(vertex=("latitude", "longitude"))
        vertex_features = []
        vertex_positions = []
        for vertex in vertices:
            # get all input vars and append lat-lon
            v_features = (
                vertices_input_vars.sel(vertex=vertex)
                .to_array(dim="variable", name=None)
                .values
            )
            v_position = [
                np.cos(vertex[0]),
                np.sin(vertex[0]),
                np.cos(vertex[1]),
                np.sin(vertex[1]),
            ]
            vertex_features.append(v_features)
            vertex_positions.append(v_position)

        grid = np.meshgrid(lat_coords, lon_coords)
        result = (grid, vertices, vertices_idx, vertex_features, vertex_positions)

        return result

    def _create_global_vertices(self, time):
        result = self._read_from_cache(key="global_{}".format(time))
        if result is not None:
            return result

        global_region = self._cube
        lat_target = len(global_region.coords["latitude"]) // self._global_scale_factor
        lon_target = len(global_region.coords["longitude"]) // self._global_scale_factor
        logger.debug("Global view dimensions = ({},{})".format(lat_target, lon_target))
        global_agg = global_region.coarsen(
            latitude=lat_target, longitude=lon_target
        ).mean(skipna=True)

        # find center_time in time coords
        time_idx = np.where(global_region["time"] == time)[0][0]
        time_slice = slice(time_idx - self._timeseries_weeks + 1, time_idx + 1)

        input_vars = self._input_vars
        if len(self._oci_input_vars) > 0:
            input_vars += self._oci_input_vars
        points_input_vars = global_agg[input_vars].isel(time=time_slice).load()

        timeseries_len = len(points_input_vars.coords["time"])
        if timeseries_len != self._timeseries_weeks:
            logger.warning(
                "Invalid time series length {} != {}".format(
                    timeseries_len, self._timeseries_weeks
                )
            )
            raise ValueError("Invalid time series length")

        # Create list of vertices
        vertices = []
        vertices_idx = {}
        for lat in global_agg.coords["latitude"].values:
            for lon in global_agg.coords["longitude"].values:
                cur_vertex = (lat, lon)
                vertices_idx[cur_vertex] = len(vertices)
                vertices.append(cur_vertex)

        # Create vertex feature tensors
        vertices_input_vars = points_input_vars.stack(vertex=("latitude", "longitude"))
        vertex_features = []
        vertex_positions = []
        for vertex in vertices:
            # get all input vars and append lat-lon
            v_features = (
                vertices_input_vars.sel(vertex=vertex)
                .to_array(dim="variable", name=None)
                .values
            )
            v_position = [
                np.cos(vertex[0]),
                np.sin(vertex[0]),
                np.cos(vertex[1]),
                np.sin(vertex[1]),
            ]
            vertex_features.append(v_features)
            vertex_positions.append(v_position)

        grid = np.meshgrid(global_agg["latitude"], global_agg["longitude"])

        result = (grid, vertices, vertices_idx, vertex_features, vertex_positions)
        self._write_to_cache(key="global_{}".format(time), data=result)

        return result

    def _create_local_edges(self, lat, lon, radius, vertices_idx):
        edges = []
        for lat_inc in range(-radius, radius + 1):
            for lon_inc in range(-radius, radius + 1):
                # vertex that we care about
                cur = (
                    lat + lat_inc * self._sp_res,
                    lon + lon_inc * self._sp_res,
                )
                cur_idx = vertices_idx[(cur[0], cur[1])]
                # logger.info("cur = {}, cur_idx={}".format(cur, cur_idx))

                # 1-hop neighbors
                cur_neighbors = self._create_all_neighbors(
                    cur, radius=3, include_self=False, normalize=True
                )
                # logger.info("cur 1-neighbors = {}".format(cur_neighbors))

                # 1-hop neighbors inside our bounding box from the center vertex
                cur_neighbors_bb = [
                    neighbor
                    for neighbor in cur_neighbors
                    if self._in_bounding_box(
                        neighbor,
                        center_lat_lon=(lat, lon),
                        radius=radius,
                    )
                ]
                cur_neighbors_bb = list(map(self._normalize_lat_lon, cur_neighbors_bb))
                cur_neighbors_bb_idx = [
                    vertices_idx[(x[0], x[1])] for x in cur_neighbors_bb
                ]
                # logger.info("cur 1-neighbors in bb = {}".format(cur_neighbors_bb))
                # logger.info("cur_idx 1-neighbors in bb = {}".format(cur_neighbors_bb_idx))

                for neighbor_idx in cur_neighbors_bb_idx:
                    # add only one direction, the other will be added by the other vertex
                    edges.append((cur_idx, neighbor_idx))
        return edges

    def _create_data(
        self,
        lat,
        lon,
        time,
        area,
        ground_truth,
        radius,
    ):
        logger.info(
            "Creating sample for lat={}, lon={}, time={}".format(
                lat, lon, time
            )
        )

        # compute local vertices
        (
            local_grid,
            local_vertices,
            local_vertices_idx,
            local_vertices_features,
            local_vertices_positions,
        ) = self._create_local_vertices(
            lat=lat,
            lon=lon,
            time=time,
            radius=radius,
        )

        # figure out index of center vertex
        center_vertex_idx = local_vertices_idx[(lat, lon)]

        # map local grid to indexes
        # local_grid_lat, local_grid_lon = local_grid
        # local_lat_lon_to_idx = np.vectorize(lambda lat, lon: local_vertices_idx[(lat, lon)])
        # local_grid_idx = local_lat_lon_to_idx(local_grid_lat, local_grid_lon)

        # compute grid edges for local vertices
        local_edges = self._create_local_edges(
            lat=lat,
            lon=lon,
            radius=radius,
            vertices_idx=local_vertices_idx,
        )

        # compute global vertices
        (
            global_grid,
            global_vertices,
            global_vertices_idx,
            global_vertices_features,
            global_vertices_positions,
        ) = self._create_global_vertices(time=time)

        # renumber global vertices based on number of local vertices
        num_local_vertices = len(local_vertices)
        global_vertices_idx = {
            k: v + num_local_vertices for k, v in global_vertices_idx.items()
        }

        # map global grid to indexes
        # global_grid_lat, global_grid_lon = global_grid
        # global_lat_lon_to_idx = np.vectorize(lambda lat, lon: global_vertices_idx[(lat, lon)])
        # global_grid_idx = global_lat_lon_to_idx(global_grid_lat, global_grid_lon)

        # link all global nodes to the central one
        global_to_local_edges = [] 
        for g_idx in global_vertices_idx.values(): 
            e = (g_idx, center_vertex_idx)
            global_to_local_edges.append(e)
            e_rev = (center_vertex_idx, g_idx)
            global_to_local_edges.append(e_rev)
        
        # logger.info("Local vertices features={}".format(local_vertices_features))
        # logger.info("Global vertices features={}".format(global_vertices_features))

        vertices_features = local_vertices_features + global_vertices_features
        vertices_features = np.array(vertices_features)
        vertices_features = torch.from_numpy(vertices_features).type(torch.float32)
        vertices_positions = local_vertices_positions + global_vertices_positions
        vertices_positions = np.array(vertices_positions)
        vertices_positions = torch.from_numpy(vertices_positions).type(torch.float32)

        graph_level_ground_truth = torch.from_numpy(np.array(ground_truth)).type(
            torch.float32
        )
        assert len(graph_level_ground_truth) == self._target_count

        area = torch.from_numpy(np.array(area)).type(torch.float32)

        # Create edge index tensor
        edges = local_edges + global_to_local_edges
        sources, targets = zip(*edges)
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        logger.debug("Computed edge tensor= {}".format(edge_index))

        data = Data(
            x=vertices_features,
            y=graph_level_ground_truth,
            edge_index=edge_index,
            pos=vertices_positions,
            area=area,
            center_lat=lat,
            center_lon=lon,
            center_time=time,
            center_vertex_idx=center_vertex_idx,
        )

        logger.debug("Computed sample={}".format(data))
        return data

    def _compute_area(self, lat, lon):
        area = self._cube["area"].sel(latitude=lat, longitude=lon)
        area_in_hectares = area.values / 10000.0
        return area_in_hectares

    def _compute_ground_truth(self, lat, lon, time):

        # find time in time coords
        time_idx = np.where(self._cube["time"] == time)[0][0]
        time_slice = slice(
            time_idx + 1,
            time_idx + 1 + self._target_count
        )
        logger.debug(
            "Computing ground truth for lat={}, lon={}, time={}, time_slice={}".format(
                lat, lon, time, time_slice
            )
        )

        target = (
            self._cube[self._target_var]
            .sel(
                latitude=lat,
                longitude=lon,
            )
            .isel(time=time_slice)
        )

        timeseries_len = len(target.coords["time"])
        if timeseries_len != self._target_count:
            logger.warning(
                "Invalid time series length {} != {}".format(
                    timeseries_len, self._target_count
                )
            )
            raise ValueError("Invalid time series length")

        logger.debug("Ground truth target values={}".format(target.values))
        return target.values

    def create(self, lat, lon, time):
        logger.debug("Creating graph for lat={}, lon={}, time={}".format(lat, lon, time))
        ground_truth = self._compute_ground_truth(
            lat, lon, time
        )
        area = self._compute_area(lat, lon)

        return self._create_data(
            lat=lat,
            lon=lon,
            time=time,
            area=area,
            ground_truth=ground_truth,
            radius=self._radius,
        )

    def _read_from_cache(self, key):
        try:
            return torch.load(
                os.path.join(self._cache_folder, "cache_item_{}.pt".format(key))
            )
        except FileNotFoundError:
            return None

    def _write_to_cache(self, key, data):
        output_path = os.path.join(self._cache_folder, "cache_item_{}.pt".format(key))
        torch.save(data, output_path)

    def _in_bounding_box(self, lat_lon, center_lat_lon, radius):
        lat, lon = lat_lon
        center_lat, center_lon = center_lat_lon
        return (
            lat <= center_lat + radius * self._sp_res
            and lat >= center_lat - radius * self._sp_res
            and lon >= center_lon - radius * self._sp_res
            and lon <= center_lon + radius * self._sp_res
        )

    def _create_all_neighbors(self, lat_lon, radius=1, include_self=False, normalize=False):
        """Create list of all neighbors inside a radius. Radius is measured in multiples of
        the spatial resolution.
        """
        lat, lon = lat_lon
        neighbors = []
        for lat_inc in range(-radius, radius + 1):
            for lon_inc in range(-radius, radius + 1):
                if not include_self and lat_inc == 0 and lon_inc == 0:
                    continue
                neighbors.append(
                    (lat + lat_inc * self._sp_res, lon + lon_inc * self._sp_res)
                )
        if normalize:
            neighbors = list(map(self._normalize_lat_lon, neighbors))
        return neighbors

    def _normalize_lat_lon(self, lat_lon):
        lat, lon = lat_lon
        while lat > self._lat_max:
            lat -= 180.0
        while lat < self._lat_min:
            lat += 180.0
        while lon < self._lon_min:
            lon += 360.0
        while lon > self._lon_max:
            lon -= 360.0
        return lat, lon

    def _normalize_lat(self, lat):
        while lat > self._lat_max:
            lat -= 180.0
        while lat < self._lat_min:
            lat += 180.0
        return lat

    def _normalize_lon(self, lon):
        while lon < self._lon_min:
            lon += 360.0
        while lon > self._lon_max:
            lon -= 360.0
        return lon

    def _create_lat_coords(self, lat, radius, normalize=True):
        lat_start = lat + radius * self._sp_res
        lat_end = lat - (radius + 1) * self._sp_res
        lat_range = np.arange(lat_start, lat_end, -self._sp_res)
        if normalize:
            lat_range = np.array(list(map(self._normalize_lat, lat_range)))
        return lat_range

    def _create_lon_coords(self, lon, radius, normalize=True):
        lon_start = lon - radius * self._sp_res
        lon_end = lon + (radius + 1) * self._sp_res
        lon_range = np.arange(lon_start, lon_end, self._sp_res)
        if normalize:
            lon_range = np.array(list(map(self._normalize_lon, lon_range)))
        return lon_range

    def _create_local_grid(self, center_lat, center_lon, radius, normalize=True):
        lat_range = self._create_lat_coords(
            lat=center_lat,
            radius=radius,
            normalize=normalize,
        )

        lon_range = self._create_lon_coords(
            lon=center_lon,
            radius=radius,
            normalize=normalize,
        )

        grid_lat, grid_lon = np.meshgrid(lat_range, lon_range)
        return grid_lat, grid_lon

