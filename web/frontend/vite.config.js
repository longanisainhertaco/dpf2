import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/token': 'http://localhost:8000',
      '/run': 'http://localhost:8000',
      '/results': 'http://localhost:8000'
    }
  }
});
