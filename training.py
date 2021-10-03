import tensorflow as tf

dim_1 = 1000 # n_sample
input_dim_1 = 365 # dimension 1 of the input
input_dim_2 = 3 # dimension 2 of the input

output_dim = 1 # dimension of the output

batch_size = 32

x_spec = tf.TensorSpec(
    shape=(n_sample, input_dim_1, input_dim_2), name="train_X", dtype=tf.dtypes.float64
)
y_spec = tf.TensorSpec(shape=(dim_1, output_dim), name="train_Y", dtype=tf.float64)

output_signature = (x_spec, y_spec)

## If you have a generator function 
# tf_train_data = tf.data.Dataset.from_generator(
#     data.generate_train_data, output_signature=output_signature
# )

# Else


tf_train_data = tf.data.Dataset.from_tensor_slices(
    (X_train, Y_train)
).batch(batch_size)


tf_val_data = tf.data.Dataset.from_tensor_slices(
    (data.val_data()[0], data.val_data()[1])
).batch(8000 * 3 - n_year + 1)

## Normalize the data

mu = []
sigma = []
for train_X, _ in tf_train_data:
    mu.append(tf.math.reduce_mean(train_X, axis=(0, 1)))
    sigma.append(tf.math.reduce_std(train_X, axis=(0, 1)))

mu = tf.reduce_mean(tf.stack(mu), axis=0)
sigma = tf.reduce_mean(tf.stack(sigma), axis=0)


def norm(x, y):
    return (x - mu) / sigma, y


train_dataset = tf_train_data.map(norm)

val_dataset = tf_val_data.map(norm)

loss_object = tf.keras.losses.MeanSquaredError()
learning_rate = 2 * 1e-1
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()


def apply_gradient(optimizer, model, x, y):

    with tf.GradientTape() as tape:
        y_pred = model(x)
        # Sample weighting becomes important for unbalanced data
        # sample_weight = np.array([0.95 if each == 1 else 0.05 for each in y])
        loss_value = loss_object(y_true=y, y_pred=y_pred)

    gradients = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    return y_pred, loss_value


def train_data_for_one_epoch():

    losses = []
    # pbar = tqdm(total=len(list(enumerate(train_dataset))), position=0, leave=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ')
    for step, (x_train, y_train) in enumerate(train_dataset):

        x_train = tf.reshape(
            x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2])
        )
        y_pred, loss_value = apply_gradient(optimizer, model, x_train, y_train)

        losses.append(loss_value)

        train_acc_metric(y_train, y_pred)

    #   pbar.set_description("Training loss for step %s: %.4f" % (int(step), float(loss_value)))
    #    pbar.update()
    return losses


def perform_validation():
    losses = []
    for x_val, y_val in val_dataset:
      
        y_pred = model(x_val)
        val_loss = loss_object(y_true=y_val, y_pred=y_pred)
        losses.append(val_loss)
        val_acc_metric(y_val, y_pred)
    return losses


epochs_val_losses, epochs_train_losses = [], []

strt_time = time.time()

for epoch in range(501):
    # Run through  training batch
    print("Start of epoch %d" % (epoch))

    losses_train = train_data_for_one_epoch()

    train_acc = train_acc_metric.result()

    losses_val = perform_validation()
    val_acc = val_acc_metric.result()

    losses_train_mean = np.mean(losses_train)
    losses_val_mean = np.mean(losses_val)
    epochs_val_losses.append(losses_val_mean)
    epochs_train_losses.append(losses_train_mean)

    fname = f"reg_linear/linear_{temporal_resolution}_{n_year}_year_lr_{learning_rate*10000}xe-4_epoch_{epoch}"
    if epoch % 10 == 0:
        fpath = MODEL_SAVE_FOLDER + fname
        model.save(fpath)

    print(
        "\n Epoch %s: Train loss: %.4f  Validation Loss: %.4f, Train PCC: %.4f, Validation PCC %.4f"
        % (
            epoch,
            float(losses_train_mean),
            float(losses_val_mean),
            float(train_acc),
            float(val_acc),
        )
    )

    train_acc_metric.reset_states()
    val_acc_metric.reset_states()

    print(f"Time taken for epoch {epoch} is {((time.time()-strt_time)/60):.2f} minutes")
