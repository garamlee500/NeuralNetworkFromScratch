from math import ceil, sqrt
from typing import Callable
import matplotlib.pyplot as plt
import torch


class Network:
    def __init__(self) -> None:
 
        self.minibatch_size = 0
        self.layers=[]

    def __repr__(self) -> str:
        return f"Neural network with {len(self.layers)} layers.\n"+\
            f"Has layers: {"".join("\n" + layer.__repr__() for layer in self.layers)}"

    def calculate(self, inputs: torch.Tensor):
        current_output = inputs
        for i in range(len(self.layers)):
            current_output = self.layers[i].calculate(current_output)
        
        return current_output

    def calculate_rectangle(
        self,
        lower_left: tuple[float, float],
        upper_right: tuple[float, float],
        density: int | tuple[int, int],
    ):
        """Evaluate the network on a 2D rectangle and cache per-layer outputs.

        Parameters
        ----------
        lower_left : tuple[float, float]
            Bottom-left corner of the rectangle as (x_min, y_min).
        upper_right : tuple[float, float]
            Top-right corner of the rectangle as (x_max, y_max).
        density : int | tuple[int, int]
            Grid resolution. If an int is provided, both axes use that many
            points. If a tuple is provided, it is interpreted as
            (x_steps, y_steps).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple (final_output_grid, xs, ys) where:
            - final_output_grid has shape (y_steps, x_steps, output_nodes)
            - xs contains sampled x coordinates
            - ys contains sampled y coordinates

        Raises
        ------
        ValueError
            If the network has no layers, does not have exactly 2 inputs in the
            first layer, or the density values are invalid.
        """
        if not self.layers:
            raise ValueError("Network has no layers.")

        if self.layers[0].input_length != 2:
            raise ValueError("Rectangle evaluation expects a network with exactly 2 inputs.")

        if isinstance(density, int):
            if density < 2:
                raise ValueError("Density must be at least 2.")
            x_steps = density
            y_steps = density
        else:
            if len(density) != 2:
                raise ValueError("Density tuple must be (x_steps, y_steps).")
            x_steps, y_steps = density
            if x_steps < 2 or y_steps < 2:
                raise ValueError("Each density value must be at least 2.")

        xs = torch.linspace(lower_left[0], upper_right[0], steps=x_steps)
        ys = torch.linspace(lower_left[1], upper_right[1], steps=y_steps)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        points = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)

        final_output = self.calculate(points)

        for layer in self.layers:
            layer.store_rectangle_outputs(xs, ys)

        return final_output.reshape(y_steps, x_steps, -1), xs, ys

    def plot_all_layer_heatmaps(
        self,
        cmap: str = "turbo",
        shared_scale: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
        robust_clip_percentile: float | None = 2.0,
        center_zero: bool = True,
    ):
        """Plot heatmaps for every node in every layer in one combined figure.

        Layers are arranged left-to-right as columns, and node indices are
        arranged top-to-bottom as rows.

        Parameters
        ----------
        cmap : str, default="turbo"
            Matplotlib colormap name used for each heatmap.
        shared_scale : bool, default=True
            If True, use one color scale across all heatmaps.
        vmin : float | None, default=None
            Optional minimum of color scale. If None, inferred from data.
        vmax : float | None, default=None
            Optional maximum of color scale. If None, inferred from data.
        robust_clip_percentile : float | None, default=2.0
            Optional tail clipping percentage used when inferring ranges.
            For example, 2.0 clips the lowest and highest 2 percent.
            Set to None or <= 0 to disable clipping.
        center_zero : bool, default=True
            If True, inferred ranges are made symmetric around zero.

        Returns
        -------
        tuple[matplotlib.figure.Figure, object]
            A tuple (fig, axes) containing the combined figure and subplot
            matrix returned by matplotlib.

        Raises
        ------
        ValueError
            If the network has no layers or rectangle outputs have not been
            cached by calling calculate_rectangle first.
        """
        if not self.layers:
            raise ValueError("Network has no layers.")

        for layer in self.layers:
            if layer.rectangle_output_grid is None or layer.rectangle_x is None or layer.rectangle_y is None:
                raise ValueError("No rectangle outputs stored for one or more layers. Run Network.calculate_rectangle(...) first.")

        layer_count = len(self.layers)
        max_nodes = max(layer.output_length for layer in self.layers)
        fig, axes = plt.subplots(max_nodes, layer_count, figsize=(4 * layer_count, 3 * max_nodes), squeeze=False)

        plot_vmin = vmin
        plot_vmax = vmax
        if shared_scale and (plot_vmin is None or plot_vmax is None):
            # Build one shared scale across all layers for direct visual comparison.
            all_values = torch.cat([layer.rectangle_output_grid.reshape(-1) for layer in self.layers])
            auto_vmin, auto_vmax = Layer._value_range(all_values, robust_clip_percentile)
            if plot_vmin is None:
                plot_vmin = auto_vmin
            if plot_vmax is None:
                plot_vmax = auto_vmax
        if shared_scale and center_zero and (plot_vmin is not None) and (plot_vmax is not None):
            plot_vmin, plot_vmax = Layer._centered_range(plot_vmin, plot_vmax)

        for layer_index, layer in enumerate(self.layers):
            output_grid = layer.rectangle_output_grid
            node_count = output_grid.shape[2]
            extent = [
                float(layer.rectangle_x[0]),
                float(layer.rectangle_x[-1]),
                float(layer.rectangle_y[0]),
                float(layer.rectangle_y[-1]),
            ]

            for node_index in range(max_nodes):
                ax = axes[node_index][layer_index]
                if node_index >= node_count:
                    ax.axis("off")
                    continue

                image = ax.imshow(
                    output_grid[:, :, node_index].cpu().numpy(),
                    origin="lower",
                    extent=extent,
                    aspect="auto",
                    cmap=cmap,
                    vmin=(
                        plot_vmin
                        if shared_scale
                        else (
                            vmin
                            if vmin is not None
                            else Layer._centered_range(*Layer._value_range(output_grid[:, :, node_index], robust_clip_percentile))[0]
                            if center_zero
                            else Layer._value_range(output_grid[:, :, node_index], robust_clip_percentile)[0]
                        )
                    ),
                    vmax=(
                        plot_vmax
                        if shared_scale
                        else (
                            vmax
                            if vmax is not None
                            else Layer._centered_range(*Layer._value_range(output_grid[:, :, node_index], robust_clip_percentile))[1]
                            if center_zero
                            else Layer._value_range(output_grid[:, :, node_index], robust_clip_percentile)[1]
                        )
                    ),
                )
                if node_index == 0:
                    ax.set_title(f"Layer {layer_index}")
                ax.set_xlabel("x")
                ax.set_ylabel(f"Node {node_index}")
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

        fig.tight_layout()
        return fig, axes

    def update_weights(self, learning_rate):
        for k in range(len(self.layers)):
            # Want to use mean gradient as this makes learning rate consistent across batch size
            self.layers[k].update_weights(learning_rate/self.minibatch_size)

            # reset gradients
            self.layers[k].gradients.zero_()
        self.minibatch_size = 0


    def calculate_with_gradient(self, inputs: torch.Tensor, target: torch.Tensor, loss_function=torch.nn.CrossEntropyLoss()):
        
        # Calculate another gradient
        self.minibatch_size+=1


        self.calculate(inputs)

        # I could derive the gradient for each loss/activation function mathematically but that wouldn't be in spirit of this project
        # (I already know how to find the gradient of a function by hand)

        yd = self.layers[-1].last_input @ self.layers[-1].weights
        yd.requires_grad= True
        loss = loss_function(self.layers[-1].activation_function(yd), target)
        loss.backward()
        dLossdyk = yd.grad


        # Process gradients for weights on each layer
        # See sgd.md for derivation

        # The input to the ith layer is x^(i) and the output is x^(i+1)
        # And has weights W^{(1)}
        # And has activation function f_{k}
        for k in range(len(self.layers)-1 , -1, -1):
            self.layers[k].gradients += torch.transpose(self.layers[k].last_input,0,1) @  dLossdyk


            # Get gradient of ReLu/activation
            # Again this would be pretty (really) straightforward to derive by hand

            # y ^((k-1))
            if k > 0:
                ykm1 = self.layers[k-1].last_input @ self.layers[k-1].weights
                ykm1.requires_grad = True
                result = self.layers[k-1].activation_function(ykm1)
                result.backward(torch.ones(ykm1.shape))
                activation_grad = torch.transpose(ykm1.grad, 0 , 1)

                # Last row of weights only affected by bias node so not in this derivative
                dLossdyk =  dLossdyk @ torch.transpose(torch.mul(activation_grad,self.layers[k].weights[:-1]), 0,1 ) 



class Layer:
    def __init__(self, input_length: int, output_length: int, activation_function: Callable[[torch.Tensor], torch.Tensor] = torch.relu):
        # self.values = torch.zeros(output_length)
        self.activation_function = activation_function
        self.input_length = input_length
        self.output_length = output_length

        
        self.weights = torch.empty(input_length+1, output_length)
        # TODO: UNDERSTAND THIS
        # (basic randomness seems sensible)
        torch.nn.init.normal_(self.weights, 0, sqrt(2/input_length))

        # Clearly I only need one of these really but these make it easier to handle
        self.last_input = torch.empty(input_length+1)
        self.last_output = torch.empty(output_length)

        # Filled by Network.calculate_rectangle for visualization.
        self.rectangle_x = None
        self.rectangle_y = None
        self.rectangle_output_grid = None

        # Store gradient (temporarily) for sgd
        # Is gradient of loss respect to weights
        self.gradients = torch.zeros(input_length+1, output_length)

    def calculate(self, prev_layer:torch.Tensor):
        if prev_layer.dim() == 1:
            prev_layer = prev_layer.unsqueeze(0)

        # Add bias node
        # Doesn't really affect maths as gradients only depend on weights
        bias_column = torch.ones((prev_layer.shape[0], 1), dtype=prev_layer.dtype, device=prev_layer.device)
        self.last_input = torch.cat((prev_layer, bias_column), 1)
        self.last_output = self.activation_function(torch.mm(self.last_input, self.weights))
        return self.last_output.clone()

    def store_rectangle_outputs(self, xs: torch.Tensor, ys: torch.Tensor):
        """Store the latest batch output as a [y, x, node] grid for plotting.

        Parameters
        ----------
        xs : torch.Tensor
            1D tensor of x coordinates used to build the rectangle grid.
        ys : torch.Tensor
            1D tensor of y coordinates used to build the rectangle grid.

        Returns
        -------
        None
            The method updates rectangle_x, rectangle_y, and
            rectangle_output_grid on the layer instance.

        Raises
        ------
        ValueError
            If last_output is not a 2D batch tensor or its batch size does not
            match len(xs) * len(ys).
        """
        if self.last_output.dim() != 2:
            raise ValueError("Layer output must be a 2D batch tensor.")

        x_steps = xs.shape[0]
        y_steps = ys.shape[0]
        expected_points = x_steps * y_steps
        if self.last_output.shape[0] != expected_points:
            raise ValueError("Layer output size does not match rectangle grid size.")

        self.rectangle_x = xs.detach().clone()
        self.rectangle_y = ys.detach().clone()
        self.rectangle_output_grid = self.last_output.detach().clone().reshape(y_steps, x_steps, self.output_length)

    def plot_rectangle_heatmaps(
        self,
        cmap: str = "turbo",
        shared_scale: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
        robust_clip_percentile: float | None = 2.0,
        center_zero: bool = True,
    ):
        """Plot one heatmap per node for this layer from cached rectangle data.

        Parameters
        ----------
        cmap : str, default="turbo"
            Matplotlib colormap name used for each heatmap.
        shared_scale : bool, default=True
            If True, use one color scale across all node heatmaps in this layer.
        vmin : float | None, default=None
            Optional minimum of color scale. If None, inferred from data.
        vmax : float | None, default=None
            Optional maximum of color scale. If None, inferred from data.
        robust_clip_percentile : float | None, default=2.0
            Optional tail clipping percentage used when inferring ranges.
            For example, 2.0 clips the lowest and highest 2 percent.
            Set to None or <= 0 to disable clipping.
        center_zero : bool, default=True
            If True, inferred ranges are made symmetric around zero.

        Returns
        -------
        tuple[matplotlib.figure.Figure, object]
            A tuple (fig, axes) containing the figure and subplot matrix.

        Raises
        ------
        ValueError
            If rectangle outputs have not been cached by calling
            Network.calculate_rectangle first.
        """
        if self.rectangle_output_grid is None or self.rectangle_x is None or self.rectangle_y is None:
            raise ValueError("No rectangle outputs stored. Run Network.calculate_rectangle(...) first.")

        output_grid = self.rectangle_output_grid
        node_count = output_grid.shape[2]
        cols = 1
        rows = node_count

        plot_vmin = vmin
        plot_vmax = vmax
        if shared_scale and (plot_vmin is None or plot_vmax is None):
            auto_vmin, auto_vmax = Layer._value_range(output_grid, robust_clip_percentile)
            if plot_vmin is None:
                plot_vmin = auto_vmin
            if plot_vmax is None:
                plot_vmax = auto_vmax
        if shared_scale and center_zero and (plot_vmin is not None) and (plot_vmax is not None):
            plot_vmin, plot_vmax = Layer._centered_range(plot_vmin, plot_vmax)

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
        extent = [
            float(self.rectangle_x[0]),
            float(self.rectangle_x[-1]),
            float(self.rectangle_y[0]),
            float(self.rectangle_y[-1]),
        ]

        for i in range(rows * cols):
            ax = axes[i // cols][i % cols]
            if i >= node_count:
                ax.axis("off")
                continue

            image = ax.imshow(
                output_grid[:, :, i].cpu().numpy(),
                origin="lower",
                extent=extent,
                aspect="auto",
                cmap=cmap,
                    vmin=(
                        plot_vmin
                        if shared_scale
                        else (
                            vmin
                            if vmin is not None
                            else Layer._centered_range(*Layer._value_range(output_grid[:, :, i], robust_clip_percentile))[0]
                            if center_zero
                            else Layer._value_range(output_grid[:, :, i], robust_clip_percentile)[0]
                        )
                    ),
                    vmax=(
                        plot_vmax
                        if shared_scale
                        else (
                            vmax
                            if vmax is not None
                            else Layer._centered_range(*Layer._value_range(output_grid[:, :, i], robust_clip_percentile))[1]
                            if center_zero
                            else Layer._value_range(output_grid[:, :, i], robust_clip_percentile)[1]
                        )
                    ),
            )
            ax.set_title(f"Node {i}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

        fig.tight_layout()
        return fig, axes


    def __repr__(self) -> str:
        return f"Layer with {self.input_length} inputs and {self.output_length} outputs.\n"+\
            f"{self.weights}"

    @staticmethod
    def _value_range(values: torch.Tensor, robust_clip_percentile: float | None):
        """Compute a min/max range, optionally clipping tails by percentile.

        Parameters
        ----------
        values : torch.Tensor
            Input tensor used to compute display limits.
        robust_clip_percentile : float | None
            Percentage of values to clip from each tail when estimating limits.
            Set to None or <= 0 to use raw min/max.

        Returns
        -------
        tuple[float, float]
            (min_value, max_value) suitable for plotting.
        """
        flattened = values.detach().reshape(-1)
        if robust_clip_percentile is None or robust_clip_percentile <= 0:
            min_value = float(flattened.min())
            max_value = float(flattened.max())
        else:
            percentile = min(max(float(robust_clip_percentile), 0.0), 49.9)
            lower_q = percentile / 100.0
            upper_q = 1.0 - lower_q
            min_value = float(torch.quantile(flattened, lower_q))
            max_value = float(torch.quantile(flattened, upper_q))

        if max_value <= min_value:
            centre = float(flattened.mean())
            epsilon = 1e-6
            return centre - epsilon, centre + epsilon

        return min_value, max_value

    @staticmethod
    def _centered_range(min_value: float, max_value: float):
        """Convert a range to a symmetric interval centered at zero.

        Parameters
        ----------
        min_value : float
            Lower bound of the original interval.
        max_value : float
            Upper bound of the original interval.

        Returns
        -------
        tuple[float, float]
            Symmetric bounds (-m, m), where m is max(abs(min_value),
            abs(max_value)).
        """
        max_abs = max(abs(min_value), abs(max_value))
        epsilon = 1e-6
        if max_abs <= 0:
            return -epsilon, epsilon
        return -max_abs, max_abs
    

    def update_weights(self, learning_rate):
        self.weights = self.weights-self.gradients*learning_rate




if __name__ == "__main__":
    xor_network = Network()
    xor_network.layers.append(Layer(2, 4, torch.sigmoid))
    xor_network.layers.append(Layer(4, 2, torch.nn.Identity()))

    inputs = torch.tensor([[[0.0,0.0]], [[0.,1.]], [[1.,0.]], [[1., 1]]])
    outputs = torch.tensor([[0], [1], [1], [0]])
    import random
    for i in range(100):
        j = random.randint(0,3)
        xor_network.calculate_with_gradient(inputs[j], outputs[j])
        xor_network.update_weights(0.1)


    print (xor_network)