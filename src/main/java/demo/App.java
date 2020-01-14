package demo;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.util.Arrays;

public class App {
  private static float[] ones(int length) {
    float[] result = new float[length];
    for (int i = 0; i < length; i++) {
      result[i] = 1.0f;
    }
    return result;
  }

  public static void main(String[] args) {
    Module mod = Module.load("train/dist/forward_scripted.pt");
    int vocab_size = 26;
    int num_ion = 12;
    int num_max_peaks = 500;
    Tensor mz_of_interest =
        Tensor.fromBlob(
            ones(vocab_size * num_ion), // data
            new long[] {1, 1, vocab_size, num_ion} // shape
            );
    Tensor spectrum_mz =
            Tensor.fromBlob(
                    ones(num_max_peaks), // data
                    new long[] {1, num_max_peaks} // shape
            );
    Tensor spectrum_intensity =
            Tensor.fromBlob(
                    ones(num_max_peaks), // data
                    new long[] {1, num_max_peaks} // shape
            );

    IValue result = mod.forward(IValue.from(mz_of_interest), IValue.from(spectrum_mz), IValue.from(spectrum_intensity));
    Tensor output = result.toTensor();
    System.out.println("shape: " + Arrays.toString(output.shape()));
    System.out.println("data: " + Arrays.toString(output.getDataAsFloatArray()));
    System.exit(0);
  }
}
