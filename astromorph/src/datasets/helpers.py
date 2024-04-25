import torch

def make_4D(image: torch.Tensor):
    """Produce a version of the image that can be run on the inference network

    Args:
        image: 2D numpy array

    Returns:
        4D numpy array that can be used for inference
    """
    # Create two extra dimensions
    image = image[None, None, :, :]
    # Create three channels per image (for RGB values)
    # Final shape is (1, 3, x, y)
    return torch.concatenate((image, image, image), 1)


def augment_image(image: torch.Tensor):
    """Create a 4D stack for image training.

    Training the model requires multiple images in a single go, because
    this is necessary for the projection in the BYOL architecture.
    Since every image has a different size, we do this by creating
    multiple copies of each image.
    These copies follow the D2 = Z2 x Z2 symmetry group.

    Args:
        image: 2D numpy array

    Returns:
        4D numpy array containing augmented copies of the original image
    """
    im_e = make_4D(image)
    im_c = torch.rot90(im_e, k=2, dims=(2, 3))
    im_b = torch.flip(im_e, dims=(2, 3))
    im_bc = torch.rot90(im_b, k=2, dims=(2, 3))

    # Concatenate along axis 0 to produce a tensor of shape (4, 3, W, H)
    images = torch.concatenate(
        (
            im_e,
            im_c,
            im_b,
            im_bc,
        ),
        0,
    )

    return images
